"""Sentiment analysis: VADER + TextBlob, with per-row agreement flag."""
from __future__ import annotations

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

from . import config
from .logger import get_logger

log = get_logger(__name__)

_VADER_POS = 0.05
_VADER_NEG = -0.05
_TB_POS = 0.1
_TB_NEG = -0.1


def _vader_label(compound: float) -> str:
    if compound >= _VADER_POS:
        return "positive"
    if compound <= _VADER_NEG:
        return "negative"
    return "neutral"


def _textblob_label(polarity: float) -> str:
    if polarity > _TB_POS:
        return "positive"
    if polarity < _TB_NEG:
        return "negative"
    return "neutral"


def score_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """Add VADER + TextBlob scores and labels. Requires `text_for_sentiment`."""
    if df.empty:
        log.warning("Empty DataFrame passed to analyzer")
        return df

    vader = SentimentIntensityAnalyzer()
    out = df.copy()

    vader_scores = out["text_for_sentiment"].apply(vader.polarity_scores)
    out["vader_compound"] = vader_scores.apply(lambda s: s["compound"])
    out["vader_label"] = out["vader_compound"].apply(_vader_label)

    out["textblob_polarity"] = out["text_for_sentiment"].apply(
        lambda t: TextBlob(t).sentiment.polarity
    )
    out["textblob_subjectivity"] = out["text_for_sentiment"].apply(
        lambda t: TextBlob(t).sentiment.subjectivity
    )
    out["textblob_label"] = out["textblob_polarity"].apply(_textblob_label)

    out["agreement"] = out["vader_label"] == out["textblob_label"]

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = config.RESULTS_DIR / "tweets_scored.csv"
    out.to_csv(target, index=False)

    agree_pct = out["agreement"].mean() * 100
    log.info("Scored %d tweets, models agree on %.1f%%", len(out), agree_pct)
    log.info("Saved scored dataset to %s", target)

    return out