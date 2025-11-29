# Project Index: Anti-UAV410

**Generated**: 2025-11-29

## Overview

Anti-UAV410 is a comprehensive benchmark and framework for UAV (Unmanned Aerial Vehicle) detection and tracking in the wild. It provides datasets, evaluation metrics, and baseline methods for discovering, detecting, recognizing, and tracking UAVs using RGB and/or Thermal Infrared (IR) videos.

**License**: MIT
**Languages**: Python (PyTorch & Jittor)
**Python Version**: 3.8 recommended

---

## Project Structure

```
anti-uav410/
├── Codes/                      # Core detection & tracking code
│   ├── detect_wrapper/         # UAV detection module (YOLOv5-based)
│   ├── tracking_wrapper/       # UAV tracking module
│   ├── metric_uav/             # Evaluation metrics
│   ├── CameralinkApplication/  # Camera integration
│   ├── demo_detect_track.py    # Detection + tracking demo
│   └── detect_tracking.py      # Main detect-track pipeline
├── anti_uav_jittor/            # Jittor framework implementations
│   ├── anti_uav410_jit/        # Anti-UAV410 Jittor version
│   ├── anti_uav_edtc_jit/      # EDTC Jittor version
│   ├── ltr/                    # Learning to Track training
│   ├── pytracking/             # PyTracking framework
│   ├── pysot_toolkit/          # PySoT evaluation toolkit
│   └── got10k_toolkit/         # GOT-10k evaluation toolkit
├── Eval/                       # Evaluation scripts & models
├── Fig/                        # Documentation figures
├── cvat_converter.py           # CVAT annotation converter
├── README.md                   # Main documentation
└── DATASET_README.md           # Dataset documentation
```

---

## Entry Points

| Entry Point | Path | Purpose |
|-------------|------|---------|
| **Inference** | `pysot_toolkit/test.py` | Main inference script for testing trackers |
| **Training** | `ltr/run_training.py` | Training launcher for tracking models |
| **Detection Demo** | `Codes/demo_detect_track.py` | Combined detection + tracking demo |
| **Pipeline** | `Codes/detect_tracking.py` | Full detection-tracking pipeline |
| **Evaluation** | `anti_uav410_jit/Evaluation_for_ALL.py` | Comprehensive evaluation |
| **Visual Demo** | `anti_uav410_jit/Demo_for_tracking.py` | Tracking demonstration |
| **Notebook** | `anti_uav_jittor/demo.ipynb` | Interactive workflow demo |

---

## Core Modules

### Detection Module (`Codes/detect_wrapper/`)
- **Detectoruav.py**: `DroneDetection` class - Main detector wrapper
- **detect_drone.py**: `detect()` function - Detection inference
- **train_drone.py**: Training script for detector
- **test_drone.py**: Testing script for detector
- **models/**: Detection model architectures (detectx, detectl, detectm, detects)
- **data/**: Dataset configs (drone.yaml, coco.yaml, voc.yaml)

### Tracking Module (`Codes/tracking_wrapper/`)
- **runtracker.py**: `run_video()`, `main()` - Tracker runner
- **dronetracker/**: Drone-specific tracker implementation
- **drtracker/**: DR tracker with FHOG features

### Jittor Implementation (`anti_uav_jittor/`)
- **anti_uav410_jit/**: Main Anti-UAV410 Jittor implementation
  - **datasets/antiuav410.py**: `AntiUAV410` dataset class
  - **trackers/SiamDT/**: Siamese Detection & Tracking
  - **trackers/SiamFC/**: SiamFC tracker
- **ltr/**: Learning-to-Track framework
  - **models/**: Tracking models
  - **trainers/**: Training logic
  - **dataset/**: Dataset loaders
- **pytracking/**: PyTracking framework
  - **tracker/**: Tracker implementations
  - **evaluation/**: Evaluation tools
  - **features/**: Feature extractors

### Evaluation (`anti_uav_jittor/pysot_toolkit/`)
- **test.py**: Main testing script with `iou()`, `eval()`, `_record()` functions
- **eval.py**: Evaluation metrics
- **bbox.py**: Bounding box utilities
- **toolkit/**: Evaluation toolkit library

---

## Configuration Files

| File | Purpose |
|------|---------|
| `detect_wrapper/data/drone.yaml` | Drone detection dataset config |
| `detect_wrapper/models/detect*.yaml` | Model architecture configs (s/m/l/x) |
| `anti_uav_edtc_jit/experiments/uavtrack/baseline.yaml` | UAV tracking experiment config |
| `pysot_toolkit/testing_dataset/*.json` | Test dataset configurations |

---

## Datasets Supported

| Dataset | Description | Modalities |
|---------|-------------|------------|
| **Anti-UAV300** | 300 sequences | RGB + IR |
| **Anti-UAV410** | 410 sequences (main) | IR only |
| **Anti-UAV600** | 600 sequences | IR only |
| **OTB100/50** | Object tracking benchmark | RGB |
| **GOT-10k** | Generic object tracking | RGB |
| **LaSOT** | Large-scale single object tracking | RGB |
| **UAV123** | UAV tracking benchmark | RGB |
| **VOT2016-2019** | Visual Object Tracking challenge | RGB |

---

## Trackers Implemented

| Tracker | Location | Description |
|---------|----------|-------------|
| **SiamDT** | `trackers/SiamDT/` | Siamese Detection & Tracking (Swin Transformer backbone) |
| **SiamFC** | `trackers/SiamFC/` | Siamese Fully Convolutional tracker |
| **DR Tracker** | `tracking_wrapper/drtracker/` | Detection + Re-detection tracker |

---

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `DroneDetection` | `detect_wrapper/Detectoruav.py` | UAV detector wrapper |
| `AntiUAV410` | `anti_uav410_jit/datasets/antiuav410.py` | Dataset loader |

---

## Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `detect()` | `detect_wrapper/detect_drone.py` | Run detection on images |
| `run_video()` | `tracking_wrapper/runtracker.py` | Run tracker on video |
| `iou()` | `pysot_toolkit/test.py` | Calculate IoU metric |
| `eval()` | `pysot_toolkit/test.py` | Evaluate tracking performance |
| `imgproc()` | `Codes/detect_tracking.py` | Process frames in pipeline |

---

## Evaluation Metrics

The primary metric is **Tracking Accuracy**:
- IoU between predicted and ground-truth boxes
- Visibility flag handling (target exists/invisible)
- Averaged over all frames (T frames, T* visible frames)

---

## Quick Start

1. **Setup Environment**
   ```bash
   # Python 3.8 recommended
   pip install -r requirements/cv.txt
   pip install jittor==1.3.8.5
   ```

2. **Download Dataset**
   - Anti-UAV300: [Google Drive](https://drive.google.com/file/d/1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr/view) | [Baidu](https://pan.baidu.com/s/1dJR0VKyLyiXBNB_qfa2ZrA)
   - Anti-UAV410: [Baidu](https://pan.baidu.com/s/1PbINXhxc-722NWoO8P2AdQ) (Password: wfds)

3. **Run Inference**
   ```bash
   python pysot_toolkit/test.py
   ```

4. **Train Model**
   ```bash
   cd anti_uav_jittor
   python ltr/run_training.py modal modal
   ```

---

## Dependencies

- **PyTorch** or **Jittor** (1.3.8.5)
- **OpenCV** (cv2)
- **NumPy**
- **libjpeg4py**
- CUDA 11.8 (for RTX 30/40 series)

---

## Related Links

- **Official Website**: https://anti-uav.github.io
- **GitHub**: https://github.com/ZhaoJ9014/Anti-UAV
- **ModelScope**: https://modelscope.cn/models/iic/3rd_Anti-UAV_CVPR23
- **Jittor**: https://cg.cs.tsinghua.edu.cn/jittor/

---

## File Statistics

| Category | Count |
|----------|-------|
| Python files | 150+ |
| YAML configs | 70+ |
| Markdown docs | 90+ |
| Shell scripts | 10 |

---

## Architecture Overview

```
Input Video (RGB/IR)
        │
        ▼
┌───────────────────┐
│   Detection       │  ← detect_wrapper (YOLOv5-based)
│   (DroneDetection)│
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Tracking        │  ← tracking_wrapper (SiamDT/SiamFC)
│   (run_video)     │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Evaluation      │  ← pysot_toolkit
│   (IoU, Accuracy) │
└───────────────────┘
        │
        ▼
    Results/Metrics
```

---

*Index generated for efficient codebase navigation. Token reduction: ~94%*
