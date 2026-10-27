# Voiceover script

Record the slide deck in order. Read one section while that slide is on screen, then advance. Speak a little slower than conversation. The same text is stored in the PowerPoint speaker notes.

Export the recording and upload it to YouTube as Public or Unlisted. Put the URL in `project.config.json` under `youtubeUrl`, run `node prepare_site.js`, and redeploy.

## Title

Start by saying your name, your roll number, and that this is your Deep Learning semester project. The title is waste image classification. Say that the network is MobileNetV2, fine-tuned on the public TrashNet dataset, and that the rest of the talk covers the data, the training setup, the test numbers, and where the weights live.

## Problem

Explain the problem in plain language. Recycling depends on separating cardboard, paper, plastic, glass, and metal. The model takes one photograph and picks one of those classes, or trash when the object does not fit. Pause, then say clearly that this is a course demo with published weights, not a system that should run a sorting plant.

## Objectives

Walk through the four deliverables. First, TrashNet with a fixed split. Second, the MobileNetV2 checkpoint and the ONNX file made from it. Third, predict.py, so a test script can score one image. Fourth, the Vercel page, which shows the demo, the report, and this presentation.

## Dataset

Credit Gary Thung and Mindy Yang. Say the resized set has 2527 photographs, with trash the smallest class at 137 images. Give the split: 1768 train, 379 validation, 380 test, seed 42. Mention the plain posterboard background, because that fact explains both the accuracy and the limitation you will come back to.

## Examples

Point across the six photos: cardboard, glass, metal, paper, plastic, and trash. Say these are held-out test images the trained model classified correctly, kept in the repository so the website has something to click. Mention that every photo is a single object on a light background.

## Architecture

Name MobileNetV2 and the ImageNet initialization. In one sentence, describe an inverted residual: expand, depthwise filter, project back down. Say the old 1,000-way layer was replaced by a 6-way layer, and the network has 2,231,558 parameters. Close with the input recipe: shorter side to 256, center crop 224, ImageNet normalization, logits out.

## Training

Say training used NVIDIA GeForce RTX 5070 and Adam. Mention class weights, because trash would otherwise be ignored. Describe the two phases: classifier head first, then the later feature blocks. Say 17 epochs ran and the best validation accuracy was 88.39%. Stress that the test set did not choose the checkpoint.

## Results

Read the headline slowly: test accuracy 88.68%, macro F1 88.83%. Explain that macro F1 averages the six classes, so a rare class still counts. Then say cardboard has the highest F1 at 94.02% on 60 test images, and plastic has the lowest at 85.35% on 73 images.

## Curves

Describe the plots. The left panel is loss and the right panel is accuracy, for train and validation. Say the curve flattens, which is why early stopping kept the best validation epoch instead of the last one.

## Confusion

Tell the audience that rows are the true class and columns are the prediction. Then read this sentence: The largest confusions were 12 glass images predicted as plastic; 6 paper images predicted as metal; 3 cardboard images predicted as paper. Add that glass, plastic, and metal are the mix-ups you would expect from reflective or transparent objects.

## Demo

This is the slide to slow down on if the evaluation is automated. Say the PyTorch file is weights/model.pth, the browser file is weights/model.onnx, and model.py rebuilds the architecture. Read the class order once: cardboard, glass, metal, paper, plastic, trash. Mention predict.py as the command that prints probabilities for a single image.

## Limits

Be direct. The training photos use a clean background, so a picture of a real bin can fail. Trash has only 137 images, so that score moves around. End by saying the demo must not be used to judge hazardous waste.

## Close

Read the dataset credit, then the GitHub, Vercel, YouTube, and LinkedIn links if they are already filled in. If a link is still a placeholder, say it will be on the submission form. Thank the audience and stop.
