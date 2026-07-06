"""Datasets for RASCAL Experiment 1.

- ``SplitArithmeticDataset`` / ``SplitArithmeticStream``: the cover task.
- ``SteganographicDataset``: wraps any base dataset with signals + trigger targets.
"""

from .math_dataset import (
    SplitArithmeticDataset,
    SplitArithmeticStream,
    ArithmeticRecord,
    sample_arithmetic,
)
from .steg_dataset import (
    SteganographicDataset,
    SteganographicStream,
    StegRecord,
    SIGNAL_MARKERS,
    RASCAL_TARGETS,
    NUM_SIGNAL_CLASSES,
    class_name,
)

__all__ = [
    "SplitArithmeticDataset",
    "SplitArithmeticStream",
    "ArithmeticRecord",
    "sample_arithmetic",
    "SteganographicDataset",
    "SteganographicStream",
    "StegRecord",
    "SIGNAL_MARKERS",
    "RASCAL_TARGETS",
    "NUM_SIGNAL_CLASSES",
    "class_name",
]
