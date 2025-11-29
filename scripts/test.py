"""Test module for YOLOv5 model evaluation.

This module re-exports the test function from evaluate.py to maintain compatibility
with train.py which imports `test` directly (i.e., `import test`).
"""

from evaluate import test

__all__ = ["test"]
