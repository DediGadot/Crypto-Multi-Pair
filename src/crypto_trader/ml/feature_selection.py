"""
Feature selection helpers for reinforcement-learning strategies.

The selector prefers XGBoost + SHAP when available, but gracefully falls back
to correlation-based ranking so the broader system keeps functioning without
GPU or proprietary dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

try:  # Heavy dependencies guarded to keep runtime lightweight when absent
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None


@dataclass
class FeatureSelectionResult:
    selected_features: List[str]
    importance: pd.DataFrame


class FeatureSelector:
    """
    Rank features using gradient-boosted trees plus SHAP explanations where possible.
    """

    def __init__(
        self,
        *,
        top_n: int = 20,
        random_state: int = 42,
    ) -> None:
        self.top_n = top_n
        self.random_state = random_state
        self._result: Optional[FeatureSelectionResult] = None

    @property
    def result(self) -> FeatureSelectionResult:
        if self._result is None:
            raise RuntimeError("FeatureSelector has not been fitted")
        return self._result

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FeatureSelectionResult:
        """
        Rank features by importance.

        Args:
            X: Feature matrix.
            y: Binary/ternary target (e.g., buy/hold/sell encoded as ints).
        """
        X_clean = X.replace([np.inf, -np.inf], np.nan).dropna()
        y_aligned = y.loc[X_clean.index]

        if len(X_clean) < 50:
            raise ValueError("Need at least 50 observations for feature selection")

        if XGBClassifier is not None:
            model = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                learning_rate=0.05,
                random_state=self.random_state,
            )
            model.fit(X_clean.values, y_aligned.values)
            importance = pd.Series(model.feature_importances_, index=X_clean.columns)

            if shap is not None:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_clean)
                importance = pd.Series(
                    np.abs(shap_values).mean(axis=0), index=X_clean.columns
                )
        else:
            # Fallback: absolute correlation with target (cheap but effective baseline).
            corr = []
            for col in X_clean.columns:
                try:
                    corr.append((col, abs(np.corrcoef(X_clean[col], y_aligned)[0, 1])))
                except Exception:
                    corr.append((col, 0.0))
            importance = pd.Series(dict(corr))

        importance = importance.sort_values(ascending=False)
        selected = importance.head(self.top_n).index.tolist()

        self._result = FeatureSelectionResult(
            selected_features=selected,
            importance=importance.reset_index().rename(
                columns={"index": "feature", 0: "importance"}
            ),
        )
        return self._result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.result.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
