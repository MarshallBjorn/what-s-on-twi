"""Text cleaning for tweets.

Produces two columns:
- text_for_sentiment: light cleaning, stop words kept (VADER needs "not"/"no")
- text_clean: aggressive cleaning, no stop words (for wordcloud/stats)
"""
from __future__ import annotations

import re
import string

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from .config import PROCESSED_DIR
from .logger import get_logger

log = get_logger(__name__)

_URL_RE = re.compile(r"http[s]?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_RT_RE = re.compile(r"^rt\s+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_WS_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_tweets(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Return df with added columns text_for_sentiment and text_clean."""
    out = df.copy()
    stops = set(stopwords.words("english"))

    light = out[text_col].astype(str).map(_light_clean)
    out["text_for_sentiment"] = light
    out["text_clean"] = light.map(lambda t: _aggressive_clean(t, stops))

    before = len(out)
    out = out[out["text_for_sentiment"].str.len() > 0].reset_index(drop=True)
    if len(out) < before:
        log.info("Dropped %d empty rows after cleaning", before - len(out))

    _persist(out)
    return out


def _light_clean(text: str) -> str:
    t = text.lower()
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = _HASHTAG_RE.sub(r"\1", t)
    t = _RT_RE.sub("", t)
    t = t.translate(_PUNCT_TABLE)
    t = _NON_ALPHA_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def _aggressive_clean(text: str, stops: set[str]) -> str:
    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if tok not in stops and len(tok) >= 2]
    return " ".join(tokens)


def _persist(df: pd.DataFrame) -> None:
    path = PROCESSED_DIR / "tweets_clean.csv"
    df.to_csv(path, index=False)
    log.info("Saved cleaned tweets to %s (%d rows)", path, len(df))