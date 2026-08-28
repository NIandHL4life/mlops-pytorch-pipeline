# MLOps PyTorch Pipeline: Docker & Kubernetes End-to-End Deployment

[![CI Pipeline](https://github.com/your-username/mlops-pytorch-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/mlops-pytorch-pipeline/actions)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?logo=pytorch)
![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED?logo=docker)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Jobs%20%26%20Deployments-326CE5?logo=kubernetes)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688?logo=fastapi)

An enterprise-grade, reproducible MLOps pipeline for training, containerizing, and orchestrating PyTorch image classification models (ResNet-18 / CIFAR-10) using Docker and Kubernetes.

---

## 📐 Architecture Overview

```mermaid
graph TD
    A[training_config.yaml / ConfigMap] -->|Mounts /app/configs| B[K8s Training Job / Dockerfile.train]
    C[(Persistent Volume /app/data)] --> B
    B -->|Logs Metrics JSON Lines| D[Stdout / Monitoring]
    B -->|Saves Checkpoint| E[(PVC /app/checkpoints/classifier_v1.pt)]
    E -->|Read-Only Mount| F[K8s Serving Deployment (2 Replicas)]
    G[FastAPI App / Dockerfile.serve] --> F
    F -->|Port 8080| H[K8s Service ClusterIP Port 80]
    I[Horizontal Pod Autoscaler HPA] -->|Autoscales 2-10 Pods| F
    J[Client / REST API] -->|POST /predict | H
    H -->|GET /health | F
```

---

## 📂 Project Structure

```text
mlops-pytorch-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI for linting, pytest, and Docker verification
├── configs/
│   └── training_config.yaml     # Hyperparameters, paths, and early stopping configs
├── docker/
│   ├── Dockerfile.train         # Optimized multi-stage build for training workloads
│   └── Dockerfile.serve         # Hardened, slim non-root container for model serving
├── k8s/
│   ├── namespace.yaml           # Isolated 'ml-training' namespace
│   ├── configmap.yaml           # ConfigMap holding training_config.yaml
│   ├── pvc.yaml                 # PersistentVolumeClaim for datasets and checkpoints
│   ├── training-job.yaml        # Batch Job manifest with resource limits & GPU toleration
│   ├── serving-deployment.yaml  # 2-replica Deployment with liveness & readiness probes
│   ├── serving-service.yaml     # Service exposing port 80 -> 8080
│   └── hpa.yaml                 # Horizontal Pod Autoscaler based on CPU metrics
├── requirements/
│   ├── train.txt                # Pinned training dependencies
│   └── serve.txt                # Pinned lightweight inference dependencies
├── src/
│   ├── __init__.py
│   ├── dataset.py               # CIFAR-10 data loaders & augmentation pipeline
│   ├── model.py                 # ResNet-18 & custom CNN model definitions
│   ├── train.py                 # Training loop with JSON metrics & early stopping
│   └── serve.py                 # High-performance FastAPI server (/health, /predict)
├── tests/
│   ├── __init__.py
│   └── test_model.py            # Unit tests for shapes, forward pass, and API
├── scripts/
│   ├── setup_git_branches.sh    # Git workflow automation script
│   ├── run_local_docker.sh      # Local Docker build and run script
│   ├── deploy_k8s.sh            # Kubernetes deployment orchestration script
│   └── test_predict.sh          # Quick prediction test script
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart: Local Development

### 1. Prerequisites
- Python 3.11+
- Docker Desktop or Docker daemon
- kubectl & Kubernetes cluster (Minikube / kind / cloud)

### 2. Python Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install training dependencies
pip install -r requirements/train.txt

# Run unit tests
pytest tests/ -v
```

### 3. Local Training
```bash
python src/train.py
```

---

## 🐳 Docker Execution

### 1. Build Multi-Stage Training Image
```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
```

### 2. Run Containerized Training with Volume Mounts
```bash
mkdir -p data checkpoints
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1
```

### 3. Build Serving Image
```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

### 4. Run Model Serving Container
```bash
docker run --rm -d \
  -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  --name mlops-serving \
  mlops-serve:v1
```

### 5. Test Serving Endpoints
```bash
# Health check
curl http://localhost:8080/health

# Prediction endpoint
curl -X POST http://localhost:8080/predict \
  -F "image=@sample_image.png"
```

---

## ☸️ Kubernetes Deployment

### 1. Deploy Namespace, ConfigMap & Storage
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
```

### 2. Run Training Job
```bash
kubectl apply -f k8s/training-job.yaml

# Follow training logs in JSON lines format
kubectl logs -f job/pytorch-training-job -n ml-training
```

### 3. Deploy Model Serving & Autoscaling
```bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# Check pod statuses
kubectl get pods -n ml-training
```

### 4. Port Forward and Test
```bash
# Port-forward service to localhost
kubectl port-forward svc/model-serving 8080:80 -n ml-training

# Test prediction
curl -X POST http://localhost:8080/predict -F "image=@sample_image.png"
```

---

## 🌿 Git & PR Workflow

This repository adheres to standard Git branching and Conventional Commits:

1. **`main`**: Production-ready code.
2. **`develop`**: Staging and feature integration branch.
3. **`feature/*`**: Scoped feature branches merged via Pull Requests.

### Merged Pull Requests:
- **PR #1 (Week 1)**: `feat(model): implement ResNet-18 model, CIFAR-10 dataset pipeline, and training loop`
- **PR #2 (Week 1)**: `feat(docker): add multi-stage Dockerfile.train and hardened Dockerfile.serve`
- **PR #3 (Week 2)**: `feat(k8s): create Kubernetes training job, configmaps, and persistent storage manifests`
- **PR #4 (Week 2)**: `feat(serving): implement Kubernetes 2-replica deployment, service, health probes, and HPA`
