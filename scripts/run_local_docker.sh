#!/usr/bin/env bash
set -e

echo "1. Building Training Docker Image (mlops-train:v1)..."
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

echo "2. Creating local storage directories..."
mkdir -p data checkpoints

echo "3. Running Training Container..."
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1

echo "4. Building Serving Docker Image (mlops-serve:v1)..."
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

echo "5. Starting Serving Container on Port 8080..."
docker run -d --name mlops-serving-app --rm \
  -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  mlops-serve:v1

echo "Waiting for serving container healthcheck..."
sleep 3
curl -f http://localhost:8080/health || echo "Waiting for server to warm up..."
echo "Ready! Test prediction with: curl -X POST http://localhost:8080/predict -F 'image=@sample.png'"
