const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

const statusEl = document.querySelector("#status");
const resultEl = document.querySelector("#result");
const classEl = document.querySelector("#pred-class");
const confEl = document.querySelector("#pred-conf");
const barsEl = document.querySelector("#bars");
const previewEl = document.querySelector("#preview");
const dropzone = document.querySelector("#dropzone");
const fileInput = document.querySelector("#file-input");

let session = null;
let classNames = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];
let previewUrl = "";

function setStatus(message) {
  statusEl.textContent = message;
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - maxLogit));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}

function resizeShorter(bitmap, shorter) {
  const scale = shorter / Math.min(bitmap.width, bitmap.height);
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(bitmap, 0, 0, width, height);
  return canvas;
}

function preprocess(bitmap) {
  const resized = resizeShorter(bitmap, 256);
  const crop = 224;
  const sx = Math.round((resized.width - crop) / 2);
  const sy = Math.round((resized.height - crop) / 2);
  const canvas = document.createElement("canvas");
  canvas.width = crop;
  canvas.height = crop;
  canvas.getContext("2d").drawImage(resized, sx, sy, crop, crop, 0, 0, crop, crop);
  const pixels = canvas.getContext("2d").getImageData(0, 0, crop, crop).data;
  const input = new Float32Array(3 * crop * crop);
  for (let index = 0; index < crop * crop; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const value = pixels[index * 4 + channel] / 255;
      input[channel * crop * crop + index] = (value - MEAN[channel]) / STD[channel];
    }
  }
  return new ort.Tensor("float32", input, [1, 3, crop, crop]);
}

function renderPrediction(probabilities) {
  const ranked = classNames
    .map((name, index) => ({ name, score: probabilities[index] }))
    .sort((a, b) => b.score - a.score);
  classEl.textContent = ranked[0].name;
  confEl.textContent = `${(ranked[0].score * 100).toFixed(2)}% confidence`;
  barsEl.innerHTML = ranked
    .map((item, index) => {
      const width = Math.max(0, Math.min(100, item.score * 100)).toFixed(2);
      return `<div class="bar-row${index === 0 ? " top" : ""}">
        <span>${item.name}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
        <strong>${width}%</strong>
      </div>`;
    })
    .join("");
  resultEl.hidden = false;
}

async function classify(bitmap) {
  if (!session) {
    setStatus("The model is still loading.");
    return;
  }
  setStatus("Running the model…");
  const tensor = preprocess(bitmap);
  const outputMap = await session.run({ input: tensor });
  const outputName = session.outputNames[0];
  const logits = Array.from(outputMap[outputName].data);
  renderPrediction(softmax(logits));
  setStatus("Prediction ready. This is an educational demo, not a waste-sorting decision.");
}

function showPreview(url) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = url;
  previewEl.src = url;
  previewEl.hidden = false;
}

async function classifyFile(file) {
  showPreview(URL.createObjectURL(file));
  const bitmap = await createImageBitmap(file);
  await classify(bitmap);
  bitmap.close();
}

async function classifyUrl(url) {
  previewEl.src = url;
  previewEl.hidden = false;
  const response = await fetch(url);
  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);
  await classify(bitmap);
  bitmap.close();
}

async function loadModel() {
  if (location.protocol === "file:") {
    setStatus("Start a local server in the web folder. Opening this file directly blocks the model download.");
    return;
  }
  const configResponse = await fetch("downloads/config.json");
  if (configResponse.ok) {
    const config = await configResponse.json();
    if (Array.isArray(config.classes) && config.classes.length) {
      classNames = config.classes;
    }
  }
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create("downloads/model.onnx", {
    executionProviders: ["wasm"],
  });
  setStatus("Model loaded. Choose a sample or upload a photo.");
}

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (file) classifyFile(file).catch((error) => setStatus(error.message));
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (file) classifyFile(file).catch((error) => setStatus(error.message));
});

document.querySelectorAll(".sample").forEach((button) => {
  button.addEventListener("click", () => {
    classifyUrl(button.dataset.src).catch((error) => setStatus(error.message));
  });
});

loadModel().catch((error) => {
  setStatus(`The model could not be loaded. ${error.message}`);
});
