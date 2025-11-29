# Training Guide: Anti-UAV YOLO Model

This guide covers training a YOLOv5 model for UAV detection using NVIDIA Brev cloud GPUs.

## Quick Start

```bash
# 1. Install dependencies with uv
uv sync

# 2. Prepare data (run locally before uploading)
uv run python scripts/prepare_data.py --input data/raw/Anti-UAV-RGBT --output data/processed --modality ir --sample-rate 5

# 3. Train (on GPU instance)
uv run python scripts/train.py --img 640 --batch 16 --epochs 100 --data data/processed/drone.yaml --weights yolov5s.pt
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Brev GPU Setup](#brev-gpu-setup)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Export](#export)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Local Machine
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- ~10GB free disk space

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Brev Account
- Sign up at [brev.nvidia.com](https://brev.nvidia.com)
- Have GPU credits available

---

## Data Preparation

### Step 1: Extract the Dataset

If not already extracted:
```bash
cd data/raw
unzip Anti-UAV-RGBT.zip
```

### Step 2: Convert to YOLO Format

```bash
# Thermal IR only (recommended for most use cases)
python scripts/prepare_data.py \
    --input data/raw/Anti-UAV-RGBT \
    --output data/processed \
    --modality ir \
    --sample-rate 5

# RGB only
python scripts/prepare_data.py \
    --input data/raw/Anti-UAV-RGBT \
    --output data/processed \
    --modality rgb \
    --sample-rate 5

# Both modalities
python scripts/prepare_data.py \
    --input data/raw/Anti-UAV-RGBT \
    --output data/processed \
    --modality both \
    --sample-rate 10
```

### Sample Rate Guidelines

| Sample Rate | ~Frames | Training Time | Accuracy |
|-------------|---------|---------------|----------|
| 1 (every frame) | 50K+ | 8+ hours | Best |
| 5 (default) | 10K | 2-3 hours | Good |
| 10 | 5K | 1-2 hours | Acceptable |
| 20 | 2.5K | 30-60 min | Quick test |

### Step 3: Verify Data

```bash
# Check image count
ls data/processed/images/train | wc -l
ls data/processed/images/val | wc -l

# Check label format (should see: 0 x_center y_center w h)
head data/processed/labels/train/*.txt
```

---

## Brev GPU Setup

### Step 1: Create Instance

1. Go to [brev.nvidia.com](https://brev.nvidia.com)
2. Click "Create Instance"
3. Select configuration:
   - **GPU**: T4 (cheapest) or A10G (faster)
   - **Image**: PyTorch 2.0 + CUDA 11.8
   - **Storage**: 50GB minimum
   - **Region**: Choose nearest for lower latency

### Step 2: Connect to Instance

```bash
# Via SSH (get command from Brev dashboard)
ssh -i ~/.ssh/your_key user@your-instance-ip

# Or use Brev CLI
brev ssh your-instance-name
```

### Step 3: Setup Environment

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/anti-uav410.git
cd anti-uav410

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (uv handles PyTorch CUDA automatically)
uv sync

# Verify GPU
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 4: Upload Data

**Option A: SCP (small datasets)**
```bash
# From local machine
scp -r data/processed user@brev-instance:~/anti-uav410/data/
```

**Option B: Rsync (large datasets, resumable)**
```bash
rsync -avz --progress data/processed/ user@brev-instance:~/anti-uav410/data/processed/
```

**Option C: Cloud Storage**
```bash
# Upload to cloud storage first, then download on instance
# Example with Google Cloud:
gsutil cp -r data/processed gs://your-bucket/anti-uav/
# On Brev instance:
gsutil cp -r gs://your-bucket/anti-uav/processed data/
```

---

## Training

### Basic Training Command

```bash
uv run python scripts/train.py \
    --img 640 \
    --batch 16 \
    --epochs 100 \
    --data data/processed/drone.yaml \
    --weights yolov5s.pt \
    --name drone_v1
```

### Training Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--img` | Input image size | 640 (standard) |
| `--batch` | Batch size | 16 (T4), 32 (A10G) |
| `--epochs` | Training epochs | 100-300 |
| `--weights` | Pretrained weights | yolov5s.pt |
| `--data` | Dataset config | data/processed/drone.yaml |
| `--name` | Experiment name | drone_v1 |
| `--device` | GPU device | 0 (single GPU) |
| `--workers` | DataLoader workers | 4-8 |

### Advanced Training

**Resume training:**
```bash
uv run python scripts/train.py --resume runs/train/drone_v1/weights/last.pt
```

**Multi-GPU (if available):**
```bash
uv run python -m torch.distributed.launch --nproc_per_node 2 scripts/train.py \
    --img 640 --batch 32 --epochs 100 --data data/processed/drone.yaml \
    --weights yolov5s.pt --device 0,1
```

**Hyperparameter tuning:**
```bash
uv run python scripts/train.py \
    --img 640 --batch 16 --epochs 100 \
    --data data/processed/drone.yaml \
    --weights yolov5s.pt \
    --hyp configs/training/finetune.yaml \
    --name drone_tuned
```

### Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir runs/train --port 6006

# In another terminal or browser, access:
# http://localhost:6006 (if running locally)
# Or use Brev port forwarding
```

### Expected Training Time

| GPU | Model | Dataset | Epochs | Time |
|-----|-------|---------|--------|------|
| T4 | YOLOv5s | 10K images | 100 | ~3 hours |
| T4 | YOLOv5m | 10K images | 100 | ~5 hours |
| A10G | YOLOv5s | 10K images | 100 | ~1.5 hours |
| A10G | YOLOv5l | 10K images | 100 | ~4 hours |

---

## Evaluation

### Evaluate on Validation Set

```bash
uv run python scripts/evaluate.py \
    --weights runs/train/drone_v1/weights/best.pt \
    --data data/processed/drone.yaml \
    --img 640 \
    --task val
```

### Evaluate on Test Set

```bash
uv run python scripts/evaluate.py \
    --weights runs/train/drone_v1/weights/best.pt \
    --data data/processed/drone.yaml \
    --img 640 \
    --task test
```

### Expected Metrics

| Metric | Target | Excellent |
|--------|--------|-----------|
| mAP@0.5 | 70-80% | 85%+ |
| mAP@0.5:0.95 | 40-50% | 55%+ |
| Precision | 80%+ | 90%+ |
| Recall | 75%+ | 85%+ |

---

## Export

### ONNX Export (for deployment)

```bash
# Install export dependencies
uv sync --extra export

uv run python scripts/export.py \
    --weights runs/train/drone_v1/weights/best.pt \
    --include onnx \
    --simplify
```

### TensorRT Export (for NVIDIA GPUs)

```bash
uv run python scripts/export.py \
    --weights runs/train/drone_v1/weights/best.pt \
    --include engine \
    --device 0
```

### Export Options

| Format | Use Case | Command |
|--------|----------|---------|
| ONNX | Cross-platform | `--include onnx` |
| TensorRT | NVIDIA GPUs | `--include engine` |
| CoreML | iOS/macOS | `--include coreml` |
| TFLite | Mobile/Edge | `--include tflite` |
| OpenVINO | Intel CPUs | `--include openvino` |

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
uv run python scripts/train.py --batch 8 ...

# Or use gradient accumulation
uv run python scripts/train.py --batch 8 --accumulate 2 ...
```

### Slow Training

```bash
# Increase workers
uv run python scripts/train.py --workers 8 ...

# Use mixed precision (faster on newer GPUs)
uv run python scripts/train.py --amp ...
```

### Poor Accuracy

1. **Check data quality**: Verify labels are correct
2. **More epochs**: Try 200-300 epochs
3. **Larger model**: Switch from YOLOv5s to YOLOv5m
4. **More data**: Reduce sample rate in data preparation
5. **Augmentation**: Enable mosaic, mixup augmentations

### Connection Lost to Brev

```bash
# Use screen or tmux for long-running jobs
screen -S training
python scripts/train.py ...
# Detach: Ctrl+A, D
# Reattach: screen -r training
```

---

## Cost Estimation

| GPU | Cost/hour | 100 epoch run | Budget |
|-----|-----------|---------------|--------|
| T4 | ~$0.50 | ~$1.50 | Tight (<10h) |
| A10G | ~$1.00 | ~$1.50 | Moderate |
| A100 | ~$3.00 | ~$1.50 | Plenty |

**Recommended for <10 hours budget:**
- Use T4 GPU
- YOLOv5s model
- 50-100 epochs
- Sample rate 5-10

---

## Next Steps

After training:
1. Download best weights: `scp user@brev:anti-uav410/runs/train/drone_v1/weights/best.pt ./`
2. Export for deployment
3. Test inference locally
4. Integrate into your application

---

## References

- [YOLOv5 Documentation](https://docs.ultralytics.com/)
- [Anti-UAV Dataset Paper](https://arxiv.org/abs/2306.15767)
- [Brev Documentation](https://docs.brev.dev/)
