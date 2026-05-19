"""Application configuration: loads .env, defines paths and constants."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
OUTPUT_DIR = PROJECT_ROOT / "output"
FALLBACK_DIR = PROJECT_ROOT / "fallback"
FALLBACK_DATASET = FALLBACK_DIR / "tweets_dataset.csv"

for _d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAPIDAPI_KEY: str = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST: str = os.getenv("RAPIDAPI_HOST", "twitter-api45.p.rapidapi.com")
SEARCH_QUERY: str = os.getenv("SEARCH_QUERY", "russia ukraine war")
TWEET_LIMIT: int = int(os.getenv("TWEET_LIMIT", "300"))