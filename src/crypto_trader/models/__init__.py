"""
Model utilities for advanced machine learning strategies.

This package centralizes reusable components used by Transformer-based
predictors and other learning-driven strategies. Modules are intentionally
lightweight and defensive so the trading system can run even when GPU training
artifacts are not present.
"""

from .datasets import SequenceDataset, build_feature_frame  # noqa: F401
from .transformer_gru import (
    TransformerGRUModel,
    load_transformer_gru,
)  # noqa: F401

__all__ = [
    "SequenceDataset",
    "build_feature_frame",
    "TransformerGRUModel",
    "load_transformer_gru",
]
