"""
Sentiment data ingestion utilities.

Designed to work with live APIs (Twitter/X, Reddit, NewsAPI) but includes
robust fallbacks so the broader system remains functional when credentials or
network access are unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from loguru import logger

try:  # Optional heavy dependencies
    import tweepy
except Exception:  # pragma: no cover
    tweepy = None

try:
    import praw
except Exception:  # pragma: no cover
    praw = None

from crypto_trader.data.storage import OHLCVStorage
from crypto_trader.features.store import FeatureStore


@dataclass
class SentimentConfig:
    twitter_accounts: Iterable[str] = ("binance", "coinbase", "krakenfx")
    reddit_subreddits: Iterable[str] = ("CryptoCurrency", "Bitcoin", "ethereum")
    news_sources: Iterable[str] = ()
    lookback_days: int = 30


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")


def load_local_sentiment_csv(symbol: str, base_dir: str | Path = "data/sentiment") -> Optional[pd.DataFrame]:
    path = Path(base_dir) / f"{_safe_symbol(symbol)}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["event_time"])
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df.sort_values("event_time")


def _fallback_from_price(symbol: str, timeframe: str, storage: Optional[OHLCVStorage] = None) -> pd.DataFrame:
    storage = storage or OHLCVStorage()
    df = storage.load_ohlcv(symbol, timeframe)
    if df is None or df.empty:
        logger.warning("Sentiment fallback unavailable - OHLCV data missing")
        return pd.DataFrame()

    df = df.copy()
    df["event_time"] = pd.to_datetime(df.index, utc=True)
    df["proxy_sentiment_score"] = df["close"].pct_change().rolling(6, min_periods=3).mean()
    df["proxy_sentiment_vol"] = df["volume"].pct_change().rolling(6, min_periods=3).std()
    df["proxy_fear_greed"] = df["proxy_sentiment_score"].rolling(24, min_periods=6).mean()
    df = df[["event_time", "proxy_sentiment_score", "proxy_sentiment_vol", "proxy_fear_greed"]].dropna()
    return df


def ingest_sentiment(
    symbol: str,
    *,
    timeframe: str = "1h",
    config: Optional[SentimentConfig] = None,
    prefer_local_csv: bool = True,
    store: Optional[FeatureStore] = None,
) -> bool:
    """
    Ingest sentiment features into the FeatureStore.
    """
    cfg = config or SentimentConfig()
    feature_store = store or FeatureStore()

    data_frame: Optional[pd.DataFrame] = None
    if prefer_local_csv:
        data_frame = load_local_sentiment_csv(symbol)

    if data_frame is None:
        try:
            data_frame = _fetch_live_sentiment(symbol, cfg)
        except Exception as exc:
            logger.warning(f"Live sentiment fetch failed: {exc}")
            data_frame = None

    if data_frame is None or data_frame.empty:
        data_frame = _fallback_from_price(symbol, timeframe)
        if data_frame.empty:
            logger.warning("Sentiment ingestion fallback produced empty frame")
            return False

    data_frame = data_frame.copy()
    data_frame["event_time"] = pd.to_datetime(data_frame["event_time"], utc=True)
    columns = {col: col if col.startswith("sent.") else f"sent.{col}" for col in data_frame.columns if col != "event_time"}
    data_frame = data_frame.rename(columns=columns)

    success = feature_store.write(data_frame, symbol=symbol, pillar="sent")
    if success:
        logger.success(f"Sentiment features ingested for {symbol}")
    return success


def _fetch_live_sentiment(symbol: str, cfg: SentimentConfig) -> Optional[pd.DataFrame]:
    """
    Attempt to fetch sentiment from available APIs.
    """
    frames = []
    twitter_df = _fetch_twitter(symbol, cfg)
    if twitter_df is not None:
        frames.append(twitter_df)

    reddit_df = _fetch_reddit(symbol, cfg)
    if reddit_df is not None:
        frames.append(reddit_df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True).sort_values("event_time")
    agg = df.groupby(pd.Grouper(key="event_time", freq="1H")).mean().reset_index()
    return agg.dropna()


def _fetch_twitter(symbol: str, cfg: SentimentConfig) -> Optional[pd.DataFrame]:
    api_key = os.getenv("TWITTER_API_KEY")
    api_secret = os.getenv("TWITTER_API_SECRET")
    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer or tweepy is None:
        return None

    client = tweepy.Client(bearer_token=bearer, consumer_key=api_key, consumer_secret=api_secret)
    rows = []
    for account in cfg.twitter_accounts:
        try:
            tweets = client.search_recent_tweets(
                query=f"from:{account} {symbol.split('/')[0]}",
                max_results=50,
                tweet_fields=["created_at", "public_metrics"],
            )
            if tweets.data is None:
                continue
            for tweet in tweets.data:
                metrics = tweet.public_metrics or {}
                rows.append(
                    {
                        "event_time": tweet.created_at,
                        "twitter_likes": metrics.get("like_count", 0),
                        "twitter_retweets": metrics.get("retweet_count", 0),
                    }
                )
        except Exception as exc:
            logger.debug(f"Twitter fetch failed for {account}: {exc}")

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df


def _fetch_reddit(symbol: str, cfg: SentimentConfig) -> Optional[pd.DataFrame]:
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "crypto-trader-bot/0.1")
    if not client_id or praw is None:
        return None

    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent, check_for_async=False)
    rows = []
    for subreddit in cfg.reddit_subreddits:
        try:
            submissions = reddit.subreddit(subreddit).new(limit=50)
            for submission in submissions:
                rows.append(
                    {
                        "event_time": pd.to_datetime(submission.created_utc, unit="s", utc=True),
                        "reddit_score": submission.score,
                        "reddit_comments": submission.num_comments,
                    }
                )
        except Exception as exc:
            logger.debug(f"Reddit fetch failed for {subreddit}: {exc}")

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    return df
