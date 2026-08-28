# Assignment 2 Submission & Reflection Report
**Course:** MLOps & Infrastructure for Machine Learning  
**Project:** End-to-End PyTorch ML Workloads with Docker & Kubernetes  
**Repository:** `mlops-pytorch-pipeline`

---

## 🔗 Submission Links
- **GitHub Repository:** `https://github.com/your-username/mlops-pytorch-pipeline`
- **Final Validation PR:** `https://github.com/your-username/mlops-pytorch-pipeline/pull/4`

---

## 📝 Reflection: Challenges & Key Architectural Learnings (350+ Words)

Deploying a machine learning workload from local scripts to containerized Kubernetes orchestration highlighted critical differences between standard web microservices and ML engineering pipelines.

### 1. Storage & State Coordination between Jobs and Deployments
The most technically challenging aspect of this project was designing the lifecycle and persistence handoff between the transient **Kubernetes Training Job** and the long-running **Model Serving Deployment**. Unlike stateless backend microservices, model training produces heavyweight binary checkpoints that must be immutably preserved and made immediately accessible to downstream serving pods. 

Configuring `ReadWriteOnce` vs. `ReadOnlyMany` access modes on PersistentVolumeClaims (PVC) required careful design: the training pod required write permissions to write the best checkpoint (`classifier_v1.pt`), whereas the serving deployment pods mounted the checkpoint directory as `readOnly: true`. In Minikube/local kind environments, ensuring the volume subpath mounts (`subPath: checkpoints`) matched between different pod specs was essential to prevent mounting collisions.

### 2. Fine-Tuning Kubernetes Health Probes for Deep Learning Models
Another major challenge was correctly calibrating Kubernetes `readinessProbe` and `livenessProbe` timing. Deep learning frameworks like PyTorch have a non-trivial startup latency while initialising CUDA/CPU device contexts, loading state dicts into memory, and executing test graph passes. If `initialDelaySeconds` is set too aggressively (e.g., < 5s), Kubernetes marks the container unready prematurely and initiates a restart loop (*CrashLoopBackOff*). By setting `initialDelaySeconds: 15` with a 5s period for the readiness probe and graceful fallback handling in FastAPI's startup event, the deployment achieved zero-downtime rolling updates with `maxUnavailable: 0` and `maxSurge: 1`.

### 3. Multi-Stage Docker Optimization and Security Hardening
Creating a lean serving container required decoupling training-heavy packages (like `pytest`, `tqdm`, and compiler toolchains) from the final runtime image. By utilizing multi-stage Docker builds and switching to a dedicated `appuser` (non-root UID 1000) with a built-in Docker `HEALTHCHECK` instruction, the serving container footprint was kept minimal and secured against root execution vulnerabilities.

---

## 📊 Summary of Pull Requests
| PR # | Branch | Title | Week |
|---|---|---|---|
| **PR #1** | `feature/model-and-training` | `feat(model): implement ResNet-18, CIFAR-10 data loader, and training loop` | Week 1 |
| **PR #2** | `feature/docker-containerization` | `feat(docker): add multi-stage Dockerfile.train and secure Dockerfile.serve` | Week 1 |
| **PR #3** | `feature/k8s-training-job` | `feat(k8s): create Kubernetes training job, configmaps, and persistent storage` | Week 2 |
| **PR #4** | `feature/k8s-serving-hpa` | `feat(k8s): implement 2-replica serving deployment, health probes, and HPA` | Week 2 |
