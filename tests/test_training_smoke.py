"""
Smoke tests for the training pipeline.

These tests verify that the training loop can:
1. Load a model
2. Load data
3. Run forward/backward passes
4. Update weights

They use minimal data and iterations to run quickly.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Add detect_wrapper to path (needed for internal imports)
DETECT_WRAPPER_PATH = PROJECT_ROOT / "Codes" / "detect_wrapper"
if str(DETECT_WRAPPER_PATH) not in sys.path:
    sys.path.insert(0, str(DETECT_WRAPPER_PATH))

# Change to detect_wrapper dir for relative imports to work
_original_cwd = os.getcwd()


@pytest.fixture(autouse=True)
def change_to_detect_wrapper():
    """Change to detect_wrapper directory for imports."""
    os.chdir(DETECT_WRAPPER_PATH)
    yield
    os.chdir(_original_cwd)


class TestModelInstantiation:
    """Tests for model creation and basic operations."""

    def test_model_from_yaml(self, device):
        """Test that model can be instantiated from YAML config."""
        from models.detect_model import Model

        # Use the small model config
        cfg_path = DETECT_WRAPPER_PATH / "models" / "detects.yaml"

        if not cfg_path.exists():
            pytest.skip(f"Model config not found: {cfg_path}")

        model = Model(str(cfg_path), ch=3, nc=1)  # 1 class for drone
        model = model.to(device)

        assert model is not None
        assert hasattr(model, "forward")

    def test_model_forward_pass(self, device, sample_image):
        """Test that model can perform forward pass."""
        from models.detect_model import Model

        cfg_path = DETECT_WRAPPER_PATH / "models" / "detects.yaml"

        if not cfg_path.exists():
            pytest.skip(f"Model config not found: {cfg_path}")

        model = Model(str(cfg_path), ch=3, nc=1)
        model = model.to(device)
        model.eval()

        # Prepare input: [batch, channels, height, width]
        # Resize image to 640x640 for YOLO
        import cv2

        img_resized = cv2.resize(sample_image, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)

        # Output should be a list of detection tensors
        assert output is not None
        assert isinstance(output, (list, tuple, torch.Tensor))

    def test_model_parameter_count(self, device):
        """Test that model has expected number of parameters."""
        from models.detect_model import Model

        cfg_path = DETECT_WRAPPER_PATH / "models" / "detects.yaml"

        if not cfg_path.exists():
            pytest.skip(f"Model config not found: {cfg_path}")

        model = Model(str(cfg_path), ch=3, nc=1)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # YOLOv5s should have ~7M parameters
        assert total_params > 1_000_000, "Model seems too small"
        assert total_params < 100_000_000, "Model seems too large"
        assert trainable_params == total_params, "All params should be trainable by default"


class TestDataLoading:
    """Tests for data loading functionality."""

    def test_create_dataloader(self, sample_dataset, hyp_config, device):
        """Test that dataloader can be created from sample dataset."""
        from utils.datasets import create_dataloader

        dataset_yaml = sample_dataset / "dataset.yaml"
        with open(dataset_yaml) as f:
            data_dict = yaml.safe_load(f)

        train_path = sample_dataset / data_dict["train"]

        # Create a simple opt-like object
        class Opt:
            single_cls = True
            rect = False
            cache_images = False
            image_weights = False
            quad = False
            world_size = 1

        opt = Opt()

        dataloader, dataset = create_dataloader(
            path=str(train_path),
            imgsz=640,
            batch_size=2,
            stride=32,
            opt=opt,
            hyp=hyp_config,
            augment=False,
            cache=False,
            rect=False,
            rank=-1,
            world_size=1,
            workers=0,  # Use 0 workers for testing
        )

        assert dataloader is not None
        assert len(dataset) > 0

        # Get one batch
        batch = next(iter(dataloader))
        assert len(batch) == 4  # imgs, targets, paths, shapes

        imgs, targets, paths, shapes = batch
        assert imgs.shape[0] <= 2  # batch size
        assert imgs.shape[1] == 3  # channels
        assert imgs.shape[2] == 640  # height
        assert imgs.shape[3] == 640  # width


class TestTrainingLoop:
    """Smoke tests for the training loop."""

    @pytest.mark.slow
    def test_single_training_iteration(self, sample_dataset, hyp_config, device):
        """Test a single forward/backward pass (smoke test)."""
        from models.detect_model import Model
        from utils.datasets import create_dataloader
        from utils.loss import ComputeLoss

        cfg_path = DETECT_WRAPPER_PATH / "models" / "detects.yaml"

        if not cfg_path.exists():
            pytest.skip(f"Model config not found: {cfg_path}")

        # Load data config
        dataset_yaml = sample_dataset / "dataset.yaml"
        with open(dataset_yaml) as f:
            data_dict = yaml.safe_load(f)

        train_path = sample_dataset / data_dict["train"]
        nc = data_dict["nc"]

        # Create model
        model = Model(str(cfg_path), ch=3, nc=nc)
        model = model.to(device)
        model.train()

        # Attach required attributes
        model.nc = nc
        model.hyp = hyp_config
        model.gr = 1.0

        # Create dataloader
        class Opt:
            single_cls = True
            rect = False
            cache_images = False
            image_weights = False
            quad = False
            world_size = 1

        opt = Opt()

        dataloader, dataset = create_dataloader(
            path=str(train_path),
            imgsz=640,
            batch_size=2,
            stride=32,
            opt=opt,
            hyp=hyp_config,
            augment=False,
            cache=False,
            rect=False,
            rank=-1,
            world_size=1,
            workers=0,
        )

        # Create loss function
        compute_loss = ComputeLoss(model)

        # Create optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        # Get one batch
        imgs, targets, paths, _ = next(iter(dataloader))
        imgs = imgs.to(device).float() / 255.0
        targets = targets.to(device)

        # Forward pass
        pred = model(imgs)

        # Compute loss
        loss, loss_items = compute_loss(pred, targets)

        # Backward pass
        loss.backward()

        # Check gradients exist
        grad_found = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                grad_found = True
                break

        assert grad_found, "No gradients computed"

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Verify loss is reasonable
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is infinite"
        assert loss.item() > 0, "Loss should be positive"

    @pytest.mark.slow
    def test_multiple_training_iterations(self, sample_dataset, hyp_config, device):
        """Test multiple training iterations to ensure stability."""
        from models.detect_model import Model
        from utils.datasets import create_dataloader
        from utils.loss import ComputeLoss

        cfg_path = DETECT_WRAPPER_PATH / "models" / "detects.yaml"

        if not cfg_path.exists():
            pytest.skip(f"Model config not found: {cfg_path}")

        dataset_yaml = sample_dataset / "dataset.yaml"
        with open(dataset_yaml) as f:
            data_dict = yaml.safe_load(f)

        train_path = sample_dataset / data_dict["train"]
        nc = data_dict["nc"]

        model = Model(str(cfg_path), ch=3, nc=nc)
        model = model.to(device)
        model.train()
        model.nc = nc
        model.hyp = hyp_config
        model.gr = 1.0

        class Opt:
            single_cls = True
            rect = False
            cache_images = False
            image_weights = False
            quad = False
            world_size = 1

        dataloader, _ = create_dataloader(
            path=str(train_path),
            imgsz=640,
            batch_size=2,
            stride=32,
            opt=Opt(),
            hyp=hyp_config,
            augment=False,
            cache=False,
            rect=False,
            rank=-1,
            world_size=1,
            workers=0,
        )

        compute_loss = ComputeLoss(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        losses = []
        num_iterations = 3

        for i, (imgs, targets, paths, _) in enumerate(dataloader):
            if i >= num_iterations:
                break

            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)

            pred = model(imgs)
            loss, _ = compute_loss(pred, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        # Verify losses are valid
        assert len(losses) > 0, "No iterations completed"
        assert all(not np.isnan(l) for l in losses), "NaN loss detected"
        assert all(not np.isinf(l) for l in losses), "Infinite loss detected"

        print(f"Losses over {len(losses)} iterations: {losses}")


class TestModelInference:
    """Tests for model inference mode."""

    def test_inference_mode(self, device, sample_image):
        """Test model in inference mode produces detections."""
        from models.detect_model import Model

        cfg_path = DETECT_WRAPPER_PATH / "models" / "detects.yaml"

        if not cfg_path.exists():
            pytest.skip(f"Model config not found: {cfg_path}")

        model = Model(str(cfg_path), ch=3, nc=1)
        model = model.to(device)
        model.eval()

        import cv2

        img_resized = cv2.resize(sample_image, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            # In inference mode, model should return predictions
            output = model(img_tensor)

        # Check output structure
        assert output is not None

    def test_batch_inference(self, device, sample_image):
        """Test model can handle batch inference."""
        from models.detect_model import Model

        cfg_path = DETECT_WRAPPER_PATH / "models" / "detects.yaml"

        if not cfg_path.exists():
            pytest.skip(f"Model config not found: {cfg_path}")

        model = Model(str(cfg_path), ch=3, nc=1)
        model = model.to(device)
        model.eval()

        import cv2

        img_resized = cv2.resize(sample_image, (640, 640))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # Create batch of 4 images
        batch = img_tensor.unsqueeze(0).repeat(4, 1, 1, 1).to(device)

        with torch.no_grad():
            output = model(batch)

        assert output is not None
