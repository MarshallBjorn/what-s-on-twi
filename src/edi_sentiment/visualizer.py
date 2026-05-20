"""Static charts saved to output/. No interactive display."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend, safe before pyplot import

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from . import config
from .logger import get_logger

log = get_logger(__name__)

_LABELS = ["negative", "neutral", "positive"]
_PALETTE = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}


def _save(fig: plt.Figure, name: str) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def plot_wordcloud(df: pd.DataFrame) -> None:
    text = " ".join(df["text_clean"].dropna().tolist())
    if not text.strip():
        log.warning("No text available for wordcloud")
        return
    wc = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        collocations=False,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word cloud — cleaned tweet text")
    _save(fig, "wordcloud.png")


def plot_distribution(df: pd.DataFrame) -> None:
    counts_v = df["vader_label"].value_counts().reindex(_LABELS, fill_value=0)
    counts_t = df["textblob_label"].value_counts().reindex(_LABELS, fill_value=0)
    plot_df = pd.DataFrame({"VADER": counts_v, "TextBlob": counts_t})

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"])
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Tweet count")
    ax.set_title("Sentiment distribution — VADER vs TextBlob")
    ax.set_xticklabels(_LABELS, rotation=0)
    ax.legend(title="Model")
    _save(fig, "distribution.png")


def plot_timeline(df: pd.DataFrame) -> None:
    if df["created_at"].isna().all():
        log.warning("No timestamps available for timeline")
        return
    series = (
        df.dropna(subset=["created_at"])
        .assign(hour=lambda x: x["created_at"].dt.floor("h"))
        .groupby(["hour", "vader_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=_LABELS, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    for label in _LABELS:
        ax.plot(series.index, series[label], label=label, color=_PALETTE[label])
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Tweets per hour")
    ax.set_title("Sentiment timeline (VADER)")
    ax.legend(title="Sentiment")
    fig.autofmt_xdate()
    _save(fig, "timeline.png")


def plot_heatmap(df: pd.DataFrame) -> None:
    if df["created_at"].isna().all():
        log.warning("No timestamps available for heatmap")
        return
    sub = df.dropna(subset=["created_at"]).copy()
    sub["hour"] = sub["created_at"].dt.hour
    sub["day"] = sub["created_at"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    pivot = (
        sub.pivot_table(index="day", columns="hour", values="id", aggfunc="count")
        .reindex(day_order)
        .reindex(columns=range(24), fill_value=0)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Tweet count"})
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Day of week")
    ax.set_title("Tweet activity heatmap")
    _save(fig, "heatmap.png")


def render_all(df: pd.DataFrame) -> None:
    plot_wordcloud(df)
    plot_distribution(df)
    plot_timeline(df)
    plot_heatmap(df)