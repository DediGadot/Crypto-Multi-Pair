"""
Multi-modal sentiment fusion strategy.

Combines price action, sentiment scores, and on-chain metrics using a simple
neural-style attention weighting. Heavy transformers are optional; when the
environment lacks HuggingFace models we fall back to deterministic scoring.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    import torch
except Exception:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None
    torch = None

from crypto_trader.strategies.base import BaseStrategy, SignalType
from crypto_trader.strategies.registry import register_strategy


@register_strategy(
    name="MultiModalSentimentFusion",
    description="Blend sentiment, on-chain, and price signals to generate trades",
    tags=["sentiment", "onchain", "fusion", "sota_2025"],
)
class MultiModalSentimentFusionStrategy(BaseStrategy):
    def __init__(self, name: str = "MultiModalSentimentFusion", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self.sentiment_model_name = "ProsusAI/finbert"
        self.buy_threshold = 0.2
        self.sell_threshold = -0.2
        self._tokenizer = None
        self._model = None

    def initialize(self, config: Dict[str, Any]) -> None:
        self.buy_threshold = float(config.get("buy_threshold", 0.2))
        self.sell_threshold = float(config.get("sell_threshold", -0.2))
        self.sentiment_model_name = config.get("model_name", self.sentiment_model_name)

        if AutoModel is not None and AutoTokenizer is not None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
                self._model = AutoModel.from_pretrained(self.sentiment_model_name)
                logger.info(f"{self.name}: loaded transformer encoder {self.sentiment_model_name}")
            except Exception as exc:
                logger.warning(f"Failed to load transformer sentiment model: {exc}")
                self._tokenizer = None
                self._model = None
        else:
            logger.info(f"{self.name}: transformers not available, using heuristic fusion")

        self._initialized = True

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "sentiment_model": self.sentiment_model_name,
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self._initialized:
            raise ValueError("Strategy not initialized")

        sentiment_cols = [c for c in data.columns if c.startswith("sent.")]
        onchain_cols = [c for c in data.columns if c.startswith("onchain.")]

        if len(sentiment_cols) == 0:
            logger.debug("No sentiment features available - emitting HOLD")
            return self._hold_frame(data)

        df = data.copy()
        df["sentiment_score"] = df[sentiment_cols].astype(float).mean(axis=1)

        if onchain_cols:
            df["onchain_score"] = df[onchain_cols].astype(float).mean(axis=1)
        else:
            df["onchain_score"] = 0.0

        df["price_momentum"] = df["close"].pct_change(6).fillna(0)
        fused = self._fuse_scores(df)

        metadata = {
            "sentiment_cols": sentiment_cols,
            "onchain_cols": onchain_cols,
            "fused_score": float(fused[-1]),
        }

        result = pd.DataFrame(
            {
                "timestamp": data.get("timestamp"),
                "signal": SignalType.HOLD.value,
                "confidence": 0.0,
                "metadata": [{} for _ in range(len(data))],
            }
        )

        last_score = fused[-1]
        # Use .at for safer scalar assignment (avoids indexer incompatibility)
        last_idx = result.index[-1]
        if last_score > self.buy_threshold:
            result.at[last_idx, "signal"] = SignalType.BUY.value
            result.at[last_idx, "confidence"] = min(0.5 + last_score, 1.0)
            result.at[last_idx, "metadata"] = metadata
        elif last_score < self.sell_threshold:
            result.at[last_idx, "signal"] = SignalType.SELL.value
            result.at[last_idx, "confidence"] = min(0.5 + abs(last_score), 1.0)
            result.at[last_idx, "metadata"] = metadata
        else:
            result.at[last_idx, "metadata"] = metadata

        return result

    def _fuse_scores(self, df: pd.DataFrame) -> np.ndarray:
        sentiment = df["sentiment_score"].to_numpy()
        onchain = df["onchain_score"].to_numpy()
        momentum = df["price_momentum"].to_numpy()

        weight_sent = 0.5
        weight_onchain = 0.3
        weight_momentum = 0.2

        if self._model is not None and self._tokenizer is not None and torch is not None:
            # Use transformer CLS embeddings as dynamic weights if text column provided
            try:
                text_col = "sent.news_headline"
                if text_col in df.columns and df[text_col].notna().any():
                    sample_text = df[text_col].dropna().astype(str).iloc[-1]
                    tokenized = self._tokenizer(sample_text, return_tensors="pt", truncation=True)
                    with torch.no_grad():
                        embeddings = self._model(**tokenized).pooler_output.cpu().numpy()
                    dynamic = np.tanh(embeddings.mean())
                    weight_sent = 0.4 + 0.2 * dynamic
                    weight_onchain = 0.3 + 0.1 * dynamic
                    weight_momentum = 1.0 - weight_sent - weight_onchain
            except Exception as exc:
                logger.debug(f"Transformer fusion fallback: {exc}")

        fused = (
            weight_sent * sentiment
            + weight_onchain * onchain
            + weight_momentum * momentum
        )
        return fused

    def _hold_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": data.get("timestamp"),
                "signal": SignalType.HOLD.value,
                "confidence": 0.0,
                "metadata": [{} for _ in range(len(data))],
            }
        )
