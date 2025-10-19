"""
Feature engineering helpers for alternative strategies.
"""

from .feature_generator import (
    FeatureGenerator,
    generate_feature_matrix,
)  # noqa: F401

__all__ = ["FeatureGenerator", "generate_feature_matrix"]
