"""Test module for YOLOv5 model evaluation.

This module re-exports the test function from evaluate.py to maintain compatibility
with train.py which imports `test` directly (i.e., `import test`).
"""

import sys
from pathlib import Path

# Add detection module to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "detection"))

from evaluate import test

__all__ = ["test"]
