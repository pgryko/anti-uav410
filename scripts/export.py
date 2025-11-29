"""Export trained YOLOv5 model to various formats.

Supports export to:
- TorchScript (.torchscript.pt)
- ONNX (.onnx)
- ONNX simplified via onnxsim (.onnx)
- CoreML (.mlmodel) - on macOS

Usage:
    python scripts/export.py --weights weights/best.pt --img-size 640 --batch-size 1
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "detection"))

from models.experimental import attempt_load
from utils.activations import Hardswish
from utils.general import check_img_size, set_logging

try:
    import models
except ImportError:
    models = None


def export_torchscript(model, img, output_path: Path) -> bool:
    """Export model to TorchScript format.

    Args:
        model: PyTorch model to export.
        img: Sample input tensor.
        output_path: Path to save the exported model.

    Returns:
        True if export successful, False otherwise.
    """
    try:
        print(f"\nStarting TorchScript export with torch {torch.__version__}...")
        ts = torch.jit.trace(model, img)
        ts.save(str(output_path))
        print(f"TorchScript export success, saved as {output_path}")
        return True
    except Exception as e:
        print(f"TorchScript export failure: {e}")
        return False


def export_onnx(model, img, output_path: Path, opset_version: int = 12) -> bool:
    """Export model to ONNX format.

    Args:
        model: PyTorch model to export.
        img: Sample input tensor.
        output_path: Path to save the exported model.
        opset_version: ONNX opset version.

    Returns:
        True if export successful, False otherwise.
    """
    try:
        import onnx

        print(f"\nStarting ONNX export with onnx {onnx.__version__}...")

        # Run model once to get output type
        y = model(img)

        torch.onnx.export(
            model,
            img,
            str(output_path),
            verbose=False,
            opset_version=opset_version,
            input_names=["images"],
            output_names=["classes", "boxes"] if y is None else ["output"],
            dynamic_axes={
                "images": {0: "batch"},
                "output": {0: "batch"},
            },
        )

        # Check exported model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"ONNX export success, saved as {output_path}")
        return True
    except ImportError:
        print("ONNX export skipped: onnx not installed. Install with: pip install onnx")
        return False
    except Exception as e:
        print(f"ONNX export failure: {e}")
        return False


def simplify_onnx(input_path: Path, output_path: Path = None) -> bool:
    """Simplify ONNX model using onnxsim.

    Args:
        input_path: Path to input ONNX model.
        output_path: Path to save simplified model (defaults to overwrite input).

    Returns:
        True if simplification successful, False otherwise.
    """
    try:
        import onnx
        import onnxsim

        print(f"\nSimplifying ONNX model with onnxsim {onnxsim.__version__}...")
        model = onnx.load(str(input_path))
        model_simplified, check = onnxsim.simplify(model)

        if not check:
            print("Warning: Simplified ONNX model validation failed")

        save_path = output_path or input_path
        onnx.save(model_simplified, str(save_path))
        print(f"ONNX simplification success, saved as {save_path}")
        return True
    except ImportError:
        print(
            "ONNX simplification skipped: onnxsim not installed. Install with: pip install onnxsim"
        )
        return False
    except Exception as e:
        print(f"ONNX simplification failure: {e}")
        return False


def export_coreml(ts_model, img, output_path: Path) -> bool:
    """Export model to CoreML format (macOS only).

    Args:
        ts_model: TorchScript model.
        img: Sample input tensor.
        output_path: Path to save the exported model.

    Returns:
        True if export successful, False otherwise.
    """
    try:
        import coremltools as ct

        print(f"\nStarting CoreML export with coremltools {ct.__version__}...")
        model = ct.convert(
            ts_model,
            inputs=[ct.ImageType(name="image", shape=img.shape, scale=1 / 255.0, bias=[0, 0, 0])],
        )
        model.save(str(output_path))
        print(f"CoreML export success, saved as {output_path}")
        return True
    except ImportError:
        print("CoreML export skipped: coremltools not installed")
        return False
    except Exception as e:
        print(f"CoreML export failure: {e}")
        return False


def export_model(
    weights: str,
    img_size: list,
    batch_size: int = 1,
    export_formats: list = None,
    opset_version: int = 12,
    simplify: bool = True,
) -> dict:
    """Export model to multiple formats.

    Args:
        weights: Path to model weights (.pt file).
        img_size: Input image size [height, width].
        batch_size: Batch size for export.
        export_formats: List of formats to export ('torchscript', 'onnx', 'coreml').
        opset_version: ONNX opset version.
        simplify: Whether to simplify ONNX model.

    Returns:
        Dictionary mapping format names to export paths (or None if failed).
    """
    if export_formats is None:
        export_formats = ["torchscript", "onnx"]

    set_logging()
    t = time.time()
    results = {}

    weights_path = Path(weights)

    # Load model
    print(f"Loading model from {weights}...")
    model = attempt_load(weights, map_location=torch.device("cpu"))

    # Validate image size
    gs = int(max(model.stride))
    img_size = [check_img_size(x, gs) for x in img_size]
    print(f"Image size: {img_size}")

    # Create sample input
    img = torch.zeros(batch_size, 3, *img_size)

    # Update model for export
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # PyTorch 1.6.0 compatibility
        if models is not None and hasattr(models, "common"):
            if isinstance(m, models.common.Conv) and isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()

    model.model[-1].export = True  # Set Detect() layer to export mode
    _ = model(img)  # Dry run

    # TorchScript export
    ts_model = None
    if "torchscript" in export_formats:
        ts_path = weights_path.with_suffix(".torchscript.pt")
        if export_torchscript(model, img, ts_path):
            results["torchscript"] = ts_path
            ts_model = torch.jit.load(str(ts_path))
        else:
            results["torchscript"] = None

    # ONNX export
    if "onnx" in export_formats:
        onnx_path = weights_path.with_suffix(".onnx")
        if export_onnx(model, img, onnx_path, opset_version):
            results["onnx"] = onnx_path
            if simplify:
                simplify_onnx(onnx_path)
        else:
            results["onnx"] = None

    # CoreML export
    if "coreml" in export_formats:
        if ts_model is None:
            print("CoreML export requires TorchScript export first")
            results["coreml"] = None
        else:
            coreml_path = weights_path.with_suffix(".mlmodel")
            if export_coreml(ts_model, img, coreml_path):
                results["coreml"] = coreml_path
            else:
                results["coreml"] = None

    print(f"\nExport complete ({time.time() - t:.2f}s)")
    print("Visualize with https://github.com/lutzroeder/netron")

    return results


def main():
    """Main entry point for export script."""
    parser = argparse.ArgumentParser(
        description="Export YOLOv5 model to various formats",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="weights path",
    )
    parser.add_argument(
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="image size [height, width]",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        type=str,
        default=["torchscript", "onnx"],
        choices=["torchscript", "onnx", "coreml"],
        help="export formats",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="disable ONNX simplification",
    )
    args = parser.parse_args()

    # Expand single image size to [height, width]
    if len(args.img_size) == 1:
        args.img_size = args.img_size * 2

    print(args)

    export_model(
        weights=args.weights,
        img_size=args.img_size,
        batch_size=args.batch_size,
        export_formats=args.formats,
        opset_version=args.opset,
        simplify=not args.no_simplify,
    )


if __name__ == "__main__":
    main()
