"""Pipeline glue: fetch → clean → score → visualize."""
from __future__ import annotations

import os

import certifi
import nltk
import pandas as pd

from . import analyzer, preproccessor, scraper, visualizer
from .logger import get_logger

log = get_logger(__name__)


def _ensure_nltk_data() -> None:
    os.environ["SSL_CERT_FILE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    for pkg in ("vader_lexicon", "stopwords", "punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)


def run_pipeline(query: str, limit: int) -> None:
    _ensure_nltk_data()

    tweets = scraper.fetch_tweets(query, limit)
    df = pd.DataFrame(tweets)
    if df.empty:
        log.error("No tweets collected, aborting")
        return

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        utc=True,
        errors="coerce",
        format="%a %b %d %H:%M:%S %z %Y",
    )

    df = preproccessor.clean_tweets(df)
    if df.empty:
        log.error("Preprocessing removed all rows, aborting")
        return

    df = analyzer.score_tweets(df)
    visualizer.render_all(df)

    log.info("Pipeline complete: %d tweets analyzed, charts in output/", len(df))