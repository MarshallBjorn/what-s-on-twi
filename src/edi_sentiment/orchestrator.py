"""End-to-end pipeline: scrape -> preprocess -> analyze -> visualize."""
from __future__ import annotations

import nltk
import pandas as pd

from . import preproccesor, scraper
from .logger import get_logger

log = get_logger(__name__)


def _ensure_nltk_data() -> None:
    import os
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where()
    for pkg in ("vader_lexicon", "stopwords", "punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)


def run_pipeline(query: str, limit: int) -> None:
    """Run all pipeline stages end-to-end."""
    _ensure_nltk_data()

    tweets = scraper.fetch_tweets(query, limit)
    df = pd.DataFrame(tweets)
    if df.empty:
        log.error("No tweets collected, aborting")
        return

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        utc=True,
        errors="coerce",
    )
    df = preproccesor.clean_tweets(df)

    # TODO Wieczór 2: analyzer.analyze_sentiment + visualizer.plot_*
    log.info("Evening 1 done: %d tweets ready for sentiment analysis", len(df))