from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort
import numpy as np

app = Flask(__name__)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Load ONNX model locally
session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])

@app.route("/")
def home():
    return "NLP API is running with ONNX Runtime!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    # Tokenize
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)

    # Run ONNX inference
    outputs = session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    })

    logits = outputs[0]
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    label = "POSITIVE" if np.argmax(probs) == 1 else "NEGATIVE"
    score = float(np.max(probs))

    return jsonify({"label": label, "score": score})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
