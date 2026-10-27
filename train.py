"""Train a MobileNetV2 waste classifier and export weights, ONNX, and metrics.

Usage (from the project virtualenv):
    python download_dataset.py
    python train.py
    python train.py --samples-only
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import ARCH, CLASSES, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, build_model

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
OUTPUTS_DIR = ROOT / "outputs"
SAMPLES_DIR = ROOT / "samples"
SEED = 42
HEAD_EPOCHS = 8
FINE_EPOCHS = 20
HEAD_PATIENCE = 3
FINE_PATIENCE = 5
BATCH_SIZE = 32


class ImageList(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = ImageOps.exif_transpose(Image.open(self.paths[index])).convert("RGB")
        return self.transform(image), self.labels[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_root(data_dir: Path) -> Path:
    for candidate in (
        data_dir / "dataset-resized",
        data_dir / "dataset-resized" / "dataset-resized",
        data_dir,
    ):
        if all((candidate / name).is_dir() for name in CLASSES):
            return candidate
    raise SystemExit("TrashNet class folders were not found. Run python download_dataset.py first.")


def collect_images(root: Path) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    for index, name in enumerate(CLASSES):
        files = sorted(
            path
            for path in (root / name).iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not files:
            raise SystemExit(f"No images found in {root / name}")
        paths.extend(files)
        labels.extend([index] * len(files))
    return paths, labels


def select_device() -> torch.device:
    if not torch.cuda.is_available():
        print("CUDA is not available. Training on CPU.")
        return torch.device("cpu")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Using {name} (sm_{major}{minor})")
    try:
        probe = torch.randn(64, 64, device="cuda")
        probe = probe @ probe
        torch.cuda.synchronize()
        del probe
    except Exception as exc:
        print(f"CUDA smoke test failed ({exc}). Training on CPU.")
        return torch.device("cpu")
    return torch.device("cuda")


def make_transforms():
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def predict_split(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels_all: list[int] = []
    preds_all: list[int] = []
    confidence_all: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            probabilities = torch.softmax(model(images), dim=1)
            chosen = probabilities.argmax(dim=1)
            preds_all.extend(chosen.cpu().tolist())
            confidence_all.extend(probabilities.max(dim=1).values.cpu().tolist())
            labels_all.extend(labels.tolist())
    return np.asarray(labels_all), np.asarray(preds_all), np.asarray(confidence_all)


def train_phase(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs: int,
    patience: int,
    phase: str,
    history: list[dict],
    best_state: dict,
    best_accuracy: float,
) -> float:
    wait = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        history.append(
            {
                "phase": phase,
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_acc, 6),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"{phase} {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_state.clear()
            best_state.update({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping in phase '{phase}' after {epoch} epochs.")
                break
    return best_accuracy


def save_plots(history: list[dict], matrix: np.ndarray) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "training_curves.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test confusion matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=140)
    plt.close(fig)


def verify_onnx(model: nn.Module, path: Path) -> None:
    import onnxruntime as ort

    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        reference = model(dummy).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(None, {session.get_inputs()[0].name: dummy.numpy()})[0]
    if not np.allclose(reference, output, rtol=1e-3, atol=1e-4):
        delta = float(np.max(np.abs(reference - output)))
        raise SystemExit(f"ONNX export does not match PyTorch (max abs diff {delta:.6f}).")
    print("ONNX export matches PyTorch logits.")


def export_onnx(model: nn.Module) -> None:
    export_model = model.cpu().eval()
    for module in export_model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    try:
        torch.onnx.export(
            export_model,
            dummy,
            WEIGHTS_DIR / "model.onnx",
            input_names=["input"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            export_model,
            dummy,
            WEIGHTS_DIR / "model.onnx",
            input_names=["input"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
        )


def copy_samples(
    test_paths: list[Path],
    test_labels: list[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray | None = None,
) -> None:
    """Copy original test files. Re-encoding can flip a borderline prediction."""
    if SAMPLES_DIR.exists():
        shutil.rmtree(SAMPLES_DIR)
    SAMPLES_DIR.mkdir(parents=True)
    best: dict[str, tuple[float, Path]] = {}
    for index, (path, label, truth, pred) in enumerate(zip(test_paths, test_labels, y_true, y_pred)):
        if int(truth) != int(pred):
            continue
        name = CLASSES[int(label)]
        score = float(confidence[index]) if confidence is not None else 1.0
        current = best.get(name)
        if current is None or score > current[0]:
            best[name] = (score, Path(path))
    for name in CLASSES:
        if name not in best:
            raise SystemExit(f"No correctly classified test image for {name}.")
        shutil.copy2(best[name][1], SAMPLES_DIR / f"{name}.jpg")


def split_dataset(paths: list[Path], labels: list[int]):
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
    )
    return (
        (list(train_paths), list(train_labels)),
        (list(val_paths), list(val_labels)),
        (list(test_paths), list(test_labels)),
    )


def per_class_counts(labels: list[int]) -> dict[str, int]:
    counts = {name: 0 for name in CLASSES}
    for label in labels:
        counts[CLASSES[label]] += 1
    return counts


def main() -> None:
    set_seed(SEED)
    device = select_device()
    image_root = find_image_root(DATA_DIR)
    paths, labels = collect_images(image_root)
    print(f"Loaded {len(paths)} images from {image_root}")

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = split_dataset(paths, labels)

    train_tf, eval_tf = make_transforms()
    train_loader = DataLoader(
        ImageList(train_paths, train_labels, train_tf),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        ImageList(val_paths, val_labels, eval_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ImageList(test_paths, test_labels, eval_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    counts = np.bincount(train_labels, minlength=len(CLASSES))
    class_weights = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(pretrained=True).to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    history: list[dict] = []
    best_state: dict = {}
    best_accuracy = -1.0
    head_optimizer = torch.optim.Adam(
        (parameter for parameter in model.classifier.parameters() if parameter.requires_grad),
        lr=1e-3,
        weight_decay=1e-4,
    )
    best_accuracy = train_phase(
        model,
        train_loader,
        val_loader,
        criterion,
        head_optimizer,
        device,
        HEAD_EPOCHS,
        HEAD_PATIENCE,
        "head",
        history,
        best_state,
        best_accuracy,
    )

    model.load_state_dict(best_state)
    model.to(device)
    blocks = list(model.features)
    fine_from = 14 if len(blocks) > 14 else max(len(blocks) - 4, 0)
    for index, block in enumerate(blocks):
        trainable = index >= fine_from
        for parameter in block.parameters():
            parameter.requires_grad = trainable
    print(f"Fine-tuning feature blocks from index {fine_from} of {len(blocks)}")
    fine_optimizer = torch.optim.Adam(
        [
            {"params": [parameter for parameter in model.features.parameters() if parameter.requires_grad], "lr": 1e-4},
            {"params": [parameter for parameter in model.classifier.parameters() if parameter.requires_grad], "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    best_accuracy = train_phase(
        model,
        train_loader,
        val_loader,
        criterion,
        fine_optimizer,
        device,
        FINE_EPOCHS,
        FINE_PATIENCE,
        "finetune",
        history,
        best_state,
        best_accuracy,
    )

    model.load_state_dict(best_state)
    model.to(device)
    y_true, y_pred, y_confidence = predict_split(model, test_loader, device)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    print(f"Test accuracy={accuracy:.4f} macro_f1={macro_f1:.4f}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in best_state.items()},
        "classes": CLASSES,
        "input_size": INPUT_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "arch": ARCH,
        "imagenet_weights": "IMAGENET1K_V1",
    }
    torch.save(checkpoint, WEIGHTS_DIR / "model.pth")

    config = {
        "framework": "pytorch",
        "architecture": ARCH,
        "weights": "weights/model.pth",
        "onnx": "weights/model.onnx",
        "task": "image-classification",
        "input_size": INPUT_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "classes": CLASSES,
        "output": "logits",
        "preprocess": "Resize the shorter side to 256, center-crop 224, then ImageNet normalize.",
        "checkpoint_keys": ["state_dict", "classes", "input_size", "mean", "std", "arch"],
    }
    (WEIGHTS_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (WEIGHTS_DIR / "class_names.json").write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")

    export_onnx(model)
    verify_onnx(model, WEIGHTS_DIR / "model.onnx")
    save_plots(history, matrix)
    copy_samples(test_paths, test_labels, y_true, y_pred, y_confidence)

    parameter_count = sum(parameter.numel() for parameter in build_model(pretrained=False).parameters())
    metrics = {
        "seed": SEED,
        "dataset": "TrashNet resized (garythung/trashnet)",
        "dataset_url": "https://github.com/garythung/trashnet",
        "class_order": CLASSES,
        "counts": {
            "total": len(paths),
            "all": per_class_counts(labels),
            "train": {"total": len(train_labels), "per_class": per_class_counts(train_labels)},
            "val": {"total": len(val_labels), "per_class": per_class_counts(val_labels)},
            "test": {"total": len(test_labels), "per_class": per_class_counts(test_labels)},
        },
        "training": {
            "architecture": ARCH,
            "pretrained": "IMAGENET1K_V1",
            "input_size": INPUT_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "weight_decay": 1e-4,
            "loss": "CrossEntropyLoss with inverse-frequency class weights from the training split",
            "class_weights": {name: round(float(value), 6) for name, value in zip(CLASSES, class_weights.cpu())},
            "scheduler": "ReduceLROnPlateau (mode=max, factor=0.5, patience=2) inside each phase",
            "augmentation": [
                "RandomResizedCrop 224 scale 0.7-1.0",
                "RandomHorizontalFlip",
                "RandomRotation 15",
                "ColorJitter 0.2/0.2/0.2/0.05",
            ],
            "phases": [
                {"name": "head", "epochs_cap": HEAD_EPOCHS, "patience": HEAD_PATIENCE, "frozen": "all feature blocks", "lr": 1e-3},
                {
                    "name": "finetune",
                    "epochs_cap": FINE_EPOCHS,
                    "patience": FINE_PATIENCE,
                    "trainable": f"ImageNet backbone; classifier head, then features from block {fine_from} onward",
                    "backbone_lr": 1e-4,
                    "head_lr": 1e-3,
                },
            ],
            "device": device.type,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "parameter_count": parameter_count,
            "epochs_ran": len(history),
            "best_val_accuracy": round(best_accuracy, 6),
        },
        "history": history,
        "test": {
            "accuracy": round(accuracy, 6),
            "macro_f1": round(macro_f1, 6),
            "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 6),
            "per_class": {
                name: {
                    "precision": round(float(report[name]["precision"]), 6),
                    "recall": round(float(report[name]["recall"]), 6),
                    "f1": round(float(report[name]["f1-score"]), 6),
                    "support": int(report[name]["support"]),
                }
                for name in CLASSES
            },
            "confusion_matrix": matrix.tolist(),
        },
    }
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUTS_DIR / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0),
        encoding="utf-8",
    )
    manifest = {
        "task": "image-classification",
        "framework": "pytorch",
        "architecture": ARCH,
        "weights": "weights/model.pth",
        "onnx": "weights/model.onnx",
        "config": "weights/config.json",
        "classes": "weights/class_names.json",
        "metrics": "outputs/metrics.json",
        "predict": "python predict.py --image <path>",
        "build_model": "model.py:build_model",
        "input_size": INPUT_SIZE,
        "class_order": CLASSES,
        "output": "logits",
    }
    (ROOT / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Saved weights/model.pth, weights/model.onnx, and outputs/metrics.json")


def refresh_samples() -> None:
    """Rewrite samples/ from the saved checkpoint without training again."""
    set_seed(SEED)
    device = select_device()
    paths, labels = collect_images(find_image_root(DATA_DIR))
    (_, _), (_, _), (test_paths, test_labels) = split_dataset(paths, labels)
    _, eval_tf = make_transforms()
    loader = DataLoader(
        ImageList(test_paths, test_labels, eval_tf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    checkpoint = torch.load(WEIGHTS_DIR / "model.pth", map_location="cpu", weights_only=False)
    model = build_model(num_classes=len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    y_true, y_pred, confidence = predict_split(model, loader, device)
    copy_samples(test_paths, test_labels, y_true, y_pred, confidence)
    print("Refreshed samples/ from the highest-confidence correct test images.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--samples-only":
        refresh_samples()
    else:
        main()
