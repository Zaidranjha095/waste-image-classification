# Waste Image Classification

Deep Learning semester project. MobileNetV2 is fine-tuned on [TrashNet](https://github.com/garythung/trashnet) to classify cardboard, glass, metal, paper, plastic, and trash.

**Test accuracy:** 88.68%  
**Macro F1:** 88.83%  
**Images:** 2527 (1768 train / 379 validation / 380 test)

## Load the weights

The checkpoint is a normal Git file at `weights/model.pth` (no Git LFS). It contains `state_dict`, `classes`, `input_size` 224, ImageNet `mean` and `std`, and `arch`.

```python
import torch
from model import build_model

checkpoint = torch.load("weights/model.pth", map_location="cpu", weights_only=False)
model = build_model(num_classes=len(checkpoint["classes"]), pretrained=False)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
```

Class order: cardboard, glass, metal, paper, plastic, trash. The network returns logits.

```bash
python predict.py --image samples/paper.jpg
```

`model_manifest.json` lists the same paths. `weights/model.onnx` is the browser copy. `weights/config.json` records preprocessing: resize the shorter side to 256, center-crop 224, ImageNet normalize.

## Project layout

| Path | Role |
| --- | --- |
| `train.py` | Stratified split, two-phase fine-tune, metrics, ONNX export |
| `model.py` | MobileNetV2 with a 6-class head |
| `predict.py` | JSON prediction for one image |
| `download_dataset.py` | Downloads the resized TrashNet zip |
| `weights/` | Checkpoint, ONNX, class names, config |
| `outputs/metrics.json` | Accuracy, F1, confusion matrix, training history |
| `report/PROJECT_REPORT.md` | Written report |
| `presentation/presentation.pptx` | Slides with speaker notes |
| `presentation/VOICEOVER_SCRIPT.md` | Text to read for the YouTube recording |
| `web/` | Static site deployed on Vercel |

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python download_dataset.py
.venv\Scripts\python train.py
.venv\Scripts\python build_docs.py
node prepare_site.js
```

The CUDA 12.8 wheel is required for an RTX 50-series GPU. `train.py` uses the CPU if CUDA cannot run a kernel. The image archive stays in `data/` and is gitignored.

## Site

`node prepare_site.js` copies the weights, plots, samples, report, and slides into `web/`. Deploy the repository on Vercel. The build command is already in `vercel.json`.

Local preview, after prepare:

```bash
npx --yes serve web -p 4173
```

Then open the demo, `report.html`, and `downloads/model.pth`.

## Dataset citation

Gary Thung and Mindy Yang. TrashNet. Stanford CS 229, 2016. https://github.com/garythung/trashnet

Student Name · Roll Number · Deep Learning
