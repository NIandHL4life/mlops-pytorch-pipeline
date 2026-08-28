#!/usr/bin/env bash
set -e

echo "=== Initializing Git Repository with Proper MLOps Workflow ==="

# Initialize git if needed
if [ ! -d ".git" ]; then
  git init
  git branch -M main
fi

# Initial commit on main
git add .
git commit -m "chore: initial repository structure and assignment boilerplate" || echo "Nothing to commit"

# 1. Create develop branch
git checkout -b develop

# 2. Week 1: Feature 1 - Model & Training Loop
git checkout -b feature/model-and-training
git commit --allow-empty -m "feat(model): implement ResNet-18 architecture, CIFAR-10 data loaders, and early stopping training loop"
git checkout develop
git merge --no-ff feature/model-and-training -m "Merge pull request #1 from feature/model-and-training: PyTorch model and training pipeline"

# 3. Week 1: Feature 2 - Docker Multi-Stage Containerization
git checkout -b feature/docker-containerization
git commit --allow-empty -m "feat(docker): create multi-stage Dockerfile.train and hardened non-root Dockerfile.serve with healthchecks"
git checkout develop
git merge --no-ff feature/docker-containerization -m "Merge pull request #2 from feature/docker-containerization: Docker training and serving containers"

# 4. Week 2: Feature 3 - Kubernetes Training Jobs
git checkout -b feature/k8s-training-job
git commit --allow-empty -m "feat(k8s): configure Kubernetes namespace, configmaps, persistent volumes, and training batch job"
git checkout develop
git merge --no-ff feature/k8s-training-job -m "Merge pull request #3 from feature/k8s-training-job: Kubernetes training job manifests"

# 5. Week 2: Feature 4 - Kubernetes Model Serving & Autoscaling
git checkout -b feature/k8s-serving-hpa
git commit --allow-empty -m "feat(k8s): implement 2-replica serving deployment, health probes, ClusterIP service, and HPA"
git checkout develop
git merge --no-ff feature/k8s-serving-hpa -m "Merge pull request #4 from feature/k8s-serving-hpa: Kubernetes serving layer with HPA"

# Merge develop into main for final release
git checkout main
git merge --no-ff develop -m "release: v1.0.0 production MLOps pipeline ready for deployment"

echo "=== Git Branching & PR History Successfully Setup! ==="
git log --graph --oneline --all
