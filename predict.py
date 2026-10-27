"""Load weights/model.pth and print class probabilities for one image.

Usage:
    python predict.py --image samples/paper.jpg
    python predict.py --image path/to/photo.jpg --weights weights/model.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from model import INPUT_SIZE, build_model


def load_checkpoint(weights: Path):
    try:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights, map_location="cpu")
    classes = checkpoint.get("classes")
    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("std", [0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(checkpoint.get("input_size", INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return model, classes, transform


def predict(image_path: Path, weights: Path) -> dict:
    model, classes, transform = load_checkpoint(weights)
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    batch = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)[0]
    pairs = sorted(
        ((classes[index], float(probabilities[index])) for index in range(len(classes))),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "image": str(image_path),
        "class": pairs[0][0],
        "confidence": round(pairs[0][1], 6),
        "probabilities": {name: round(score, 6) for name, score in pairs},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a waste class for one image.")
    parser.add_argument("--image", required=True, help="Path to an RGB image.")
    parser.add_argument("--weights", default="weights/model.pth", help="Checkpoint saved by train.py.")
    args = parser.parse_args()
    result = predict(Path(args.image), Path(args.weights))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
