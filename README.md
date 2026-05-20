# What's On Twi — Analiza wydźwięku tweetów
> Eksploracja Danych Internetowych

Pobiera tweety o wojnie ros-ukr, czyści, analizuje sentyment (VADER + TextBlob), wizualizuje wyniki.

## Instalacja

```bash
uv sync
cp .env.example .env  # i wpisz swój RAPIDAPI_KEY
```

## Stack
- Python 3.11+, `uv`
- RapidAPI (`twitter-api45`) + opcjonalny fallback CSV
- VADER (NLTK) + TextBlob
- matplotlib, seaborn, wordcloud

## Uruchomienie

```bash
uv run python -m edi_sentiment
uv run python -m edi_sentiment --query "russia ukraine war" --limit 300
```

## Artefakty

- `data/raw/tweets_<ts>.json` — surowe tweety
- `data/processed/tweets_clean.csv` — po czyszczeniu
- `data/results/tweets_scored.csv` — po analizie (Wieczór 2)
- `output/*.png` — wykresy (Wieczór 2)