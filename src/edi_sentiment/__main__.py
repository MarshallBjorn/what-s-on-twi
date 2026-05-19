"""CLI entry point: python -m edi_sentiment ..."""
from __future__ import annotations

import argparse

from .config import SEARCH_QUERY, TWEET_LIMIT
from .orchestrator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="edi_sentiment")
    parser.add_argument("--query", default=SEARCH_QUERY,
                        help="Tweet search query (default from .env)")
    parser.add_argument("--limit", type=int, default=TWEET_LIMIT,
                        help="Max tweets to fetch (default from .env)")
    args = parser.parse_args()
    run_pipeline(args.query, args.limit)


if __name__ == "__main__":
    main()