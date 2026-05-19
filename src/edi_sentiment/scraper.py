"""Tweet fetching via RapidAPI with CSV fallback."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import FALLBACK_DATASET, RAPIDAPI_HOST, RAPIDAPI_KEY, RAW_DIR
from .logger import get_logger

log = get_logger(__name__)

_API_URL = f"https://{RAPIDAPI_HOST}/search.php"
_MIN_ACCEPTABLE = 50
_REQUIRED_COLS = ("id", "created_at", "text", "author", "lang")


def fetch_tweets(query: str, limit: int = 300) -> list[dict[str, Any]]:
    """Return list of dicts with keys: id, created_at, text, author, lang.
    Falls back to a local CSV when the API is unavailable or yields too few rows.
    """
    log.info("Fetching tweets: query=%r, limit=%d", query, limit)

    tweets: list[dict[str, Any]] = []
    try:
        tweets = _fetch_from_rapidapi(query, limit)
    except requests.RequestException as exc:
        log.warning("API failed (%s), switching to fallback", exc)

    threshold = min(limit, _MIN_ACCEPTABLE)
    if len(tweets) < threshold:
        if tweets:
            log.warning(
                "Only %d tweets from API (<%d), switching to fallback",
                len(tweets), threshold,
            )
        try:
            tweets = _load_fallback(limit)
        except FileNotFoundError as exc:
            log.warning("%s — keeping %d API tweets", exc, len(tweets))

    path = _persist_raw(tweets)
    log.info("Fetched %d tweets, saved to %s", len(tweets), path)
    return tweets


def _fetch_from_rapidapi(query: str, limit: int) -> list[dict[str, Any]]:
    if not RAPIDAPI_KEY:
        log.warning("RAPIDAPI_KEY not set, skipping API call")
        return []

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }
    collected: list[dict[str, Any]] = []
    cursor: str | None = None
    page = 0

    while len(collected) < limit and page < 50:
        params: dict[str, Any] = {"query": query, "search_type": "Latest"}
        if cursor:
            params["cursor"] = cursor

        resp = requests.get(_API_URL, headers=headers, params=params, timeout=30)
        if resp.status_code == 429:
            log.warning("Rate limit hit (429), aborting API path")
            break
        resp.raise_for_status()
        payload = resp.json()

        page_tweets = _extract_tweets(payload)
        if not page_tweets:
            break

        for t in page_tweets:
            if t.get("lang") == "en":
                collected.append(t)
                if len(collected) >= limit:
                    break

        cursor = payload.get("next_cursor") or payload.get("cursor")
        page += 1
        if not cursor:
            break
        time.sleep(0.3)

    return collected


def _extract_tweets(payload: dict[str, Any]) -> list[dict[str, Any]]:
    # twitter-api45 zwraca listę pod różnymi kluczami zależnie od endpointu
    raw = payload.get("timeline") or payload.get("tweets") or payload.get("data") or []
    out: list[dict[str, Any]] = []
    for item in raw:
        text = item.get("text") or item.get("full_text") or ""
        if not text:
            continue
        author = (
            (item.get("user_info") or {}).get("screen_name")
            or item.get("screen_name")
            or (item.get("author") or {}).get("screen_name")
            or ""
        )
        out.append({
            "id": str(item.get("tweet_id") or item.get("id") or ""),
            "created_at": item.get("created_at") or item.get("date"),
            "text": text,
            "author": author,
            "lang": item.get("lang") or "en",
        })
    return out


def _load_fallback(limit: int) -> list[dict[str, Any]]:
    if not FALLBACK_DATASET.exists():
        raise FileNotFoundError(
            f"Fallback dataset not found at {FALLBACK_DATASET}. "
            "Add a CSV with columns: id, created_at, text, author, lang."
        )
    df = pd.read_csv(FALLBACK_DATASET).head(limit)
    for col in _REQUIRED_COLS:
        if col not in df.columns:
            df[col] = "en" if col == "lang" else ""
    log.info("Loaded %d tweets from fallback dataset", len(df))
    return df[list(_REQUIRED_COLS)].to_dict(orient="records")


def _persist_raw(tweets: list[dict[str, Any]]) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RAW_DIR / f"tweets_{ts}.json"
    path.write_text(json.dumps(tweets, ensure_ascii=False, indent=2))
    return path