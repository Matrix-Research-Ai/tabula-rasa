# Cloud Deployment Guide

Tabula Rasa can run on any cloud provider. The minimal hardware requirement
is 4 CPU cores and 4 GB RAM — satisfied by most free-tier instances.

## Quick Comparison

| Provider | Cheapest GPU | Cost/hr | Best For |
|----------|-------------|---------|----------|
| **RunPod** | RTX 4080 | ~$0.30 | Fast training, 1-click PyTorch |
| **Vast.ai** | RTX 4080 | ~$0.20 | Cheapest GPU, auction pricing |
| **AWS** | T4 (g4dn.xlarge) | ~$0.53 | Enterprise, existing infrastructure |
| **GCP** | T4 (n1-standard-4 + T4) | ~$0.57 | GKE integration |
| **Azure** | T4 (Standard NC4as) | ~$0.62 | Azure ML integration |
| **CPU-only** | None | ~$0.01-0.05 | Free tiers, prototyping |

---

## Option 1: RunPod (Easiest, ~2 min)

[RunPod.io](https://www.runpod.io) is the fastest way to get a GPU.

```bash
# 1. Create a pod: GPU → RTX 4080 → PyTorch 2.x template → Start
# 2. Upload project zip via file manager or git clone
# 3. Install and run:
cd /root
pip install -e .
python3 train_specialist.py add --quick
```

**Cost:** ~$0.30/hr for RTX 4080. A 30K-step training run finishes in ~2 min
and costs ~$0.01.

**Cheaper:** Use RTX 3060 (~$0.10/hr) — batch=256 still achieves 1-digit
mastery in ~6 min.

---

## Option 2: Vast.ai (Cheapest GPU)

[Vast.ai](https://vast.ai) offers auction-priced GPUs. Same setup as RunPod
but typically 30-40% cheaper.

```bash
# After renting and connecting via SSH:
pip install -e .
python3 train_specialist.py add --quick
```

**Tip:** Filter by "Verified" datacenters for reliable connectivity.

---

## Option 3: AWS EC2 (Enterprise)

### GPU Instance (g4dn.xlarge — T4 GPU)

```bash
# Launch: AWS Console → EC2 → g4dn.xlarge → Deep Learning AMI (Ubuntu)

# Install project
git clone https://github.com/tabula-rasa-ai/tabula-rasa.git
cd tabula-rasa
pip install -e .
python3 train_specialist.py add --amp --steps 30000
```

### CPU-Only (t3.large — Free Tier Eligible)

```bash
# Launch: t3.large → Ubuntu 22.04
# SSH in:
sudo apt update && sudo apt install -y python3-pip git
git clone https://github.com/tabula-rasa-ai/tabula-rasa.git
cd tabula-rasa
pip install -e .
python3 train_specialist.py add --quick
# Expected: ~30s, ~100% accuracy on 1-digit addition
```

**Cost:** t3.large is ~$0.08/hr. Free tier may cover it depending on usage.

---

## Option 4: Google Cloud (GKE)

### Build and push
```bash
docker build -t tabula-rasa:latest .
docker tag tabula-rasa:latest gcr.io/$PROJECT/tabula-rasa:latest
docker push gcr.io/$PROJECT/tabula-rasa:latest
```

### Deploy to GKE
```bash
# Create cluster
gcloud container clusters create tabula-rasa --num-nodes=1 --machine-type=e2-standard-4

# Deploy
kubectl apply -f k8s/deployment.yaml

# Expose
kubectl expose deployment tabula-rasa-api --type=LoadBalancer --port=8000

# Get URL
kubectl get service tabula-rasa-api
```

---

## Option 5: Docker Compose (Single Machine)

### CPU Profile (default, no GPU)
```bash
docker compose --profile cpu up -d
```

### GPU Profile (requires nvidia-docker)
```bash
docker compose --profile gpu up -d
```

### Minimal Profile (API only)
```bash
docker compose --profile minimal up -d
```

See [docker-compose.yml](docker-compose.yml) for all services.

---

## Hardware Notes

### Minimum Requirements

| Mode | CPU | RAM | Disk | GPU |
|------|-----|-----|------|-----|
| API server only | 2 cores | 2 GB | 1 GB | None |
| Training (1M model) | 4 cores | 4 GB | 2 GB | Optional |
| Training (5M model) | 4 cores | 8 GB | 5 GB | Recommended (2 GB+ VRAM) |
| Full stack | 4 cores | 8 GB | 10 GB | Optional |

### Bottlenecks

| Component | CPU | GPU | RAM | Disk I/O |
|-----------|-----|-----|-----|----------|
| Training loop | Medium | High (if available) | Low | Low |
| EWC consolidation | High | Low | Medium | Medium |
| Sleep cycle | High | Medium | Medium | High (SQLite) |
| MCTS search | High | Medium | Medium | Low |
| API inference | Low | Low (CPU is fine) | Low | Low |
| Dashboard | Low | None | Low | Low |

### Recommended Configurations

**Development laptop (CPU-only):**
```bash
docker compose --profile minimal up -d
```

**GPU training server:**
```bash
docker compose --profile gpu up -d
```

**Kubernetes cluster:**
```bash
kubectl apply -f k8s/deployment.yaml
```

---

## Container Image Tags

| Tag | Base | PyTorch | Size | Use Case |
|-----|------|---------|------|----------|
| `latest` | python:3.11-slim | CPU | ~1.2 GB | Default, CPU inference |
| `gpu` | pytorch/pytorch:2.6.0-cuda12.4 | CUDA 12.4 | ~6 GB | GPU training |

```bash
docker build -t tabula-rasa:latest -f Dockerfile .
docker build -t tabula-rasa:gpu -f Dockerfile.gpu .
```
