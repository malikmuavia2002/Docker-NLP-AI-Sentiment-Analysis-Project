# Docker-NLP-AI-Sentiment-Analysis-Project
This project demonstrates a sentiment analysis API using Flask and ONNX Runtime. The API classifies input text as POSITIVE or NEGATIVE with a confidence score. The main goal is to provide a lightweight, fast, and deployable NLP model for text sentiment prediction that can run efficiently in a Docker container.

Key features:

Uses DistilBERT fine-tuned on SST-2 for sentiment classification.

Runs inference using ONNX Runtime for faster performance compared to standard PyTorch models.

Provides a REST API with /predict endpoint for real-time predictions.

Fully Dockerized, making deployment on any environment seamless.

Files

app.py – Flask API code that loads tokenizer and ONNX model, handles requests, and returns predictions.

model.onnx – Pre-exported ONNX model for sentiment analysis.

Dockerfile – Builds the Docker image to run the NLP API.

Requirement.txt - Includes Necessary Pakages

1. Build Docker image:
docker build -t docker-nlp-api .

2. Run container

docker run -p 8000:8000 docker-nlp-api

3. Test the NLP Prediction Endpoint
   # Send request and format JSON nicely
(curl -Method POST -Uri "http://localhost:8000/predict" -ContentType "application/json" -Body '{"text":"I love AI"}').Content | ConvertFrom-Json | Format-List

Output

<img width="827" height="122" alt="image" src="https://github.com/user-attachments/assets/25bd9d14-fd22-4936-bcf6-def8c010c763" />

<img width="854" height="179" alt="image" src="https://github.com/user-attachments/assets/3e3df7c6-ae28-412c-97b1-c1bae7631ec4" />



   
