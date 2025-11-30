#!/usr/bin/env python3
"""Simple training wrapper for Anti-UAV detection.

This script runs training from the Codes/detect_wrapper directory which has
working YOLOv5 implementation.
"""

import os
import subprocess
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent
CODES_DIR = ROOT / "Codes" / "detect_wrapper"

# Default parameters for a short test run
DEFAULT_ARGS = {
    "--data": str(ROOT / "data" / "processed" / "drone.yaml"),
    "--cfg": str(CODES_DIR / "models" / "detects.yaml"),  # YOLOv5s (small)
    "--weights": "",  # Train from scratch
    "--epochs": "3",
    "--batch-size": "8",  # Small batch for 4GB VRAM
    "--img-size": "416",  # Smaller image for memory
    "--device": "0",
    "--workers": "4",
    "--name": "drone_test",
}


def main():
    # Change to Codes/detect_wrapper directory
    os.chdir(CODES_DIR)

    # Build command
    cmd = [sys.executable, "train_drone.py"]

    # Add default args (can be overridden by CLI args)
    args_dict = DEFAULT_ARGS.copy()

    # Parse CLI args to override defaults
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith("--"):
            key = sys.argv[i]
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                args_dict[key] = sys.argv[i + 1]
                i += 2
            else:
                args_dict[key] = ""
                i += 1
        else:
            i += 1

    # Build final command
    for key, value in args_dict.items():
        cmd.append(key)
        if value:
            cmd.append(value)

    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {CODES_DIR}")
    print("-" * 60)

    # Run training
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
