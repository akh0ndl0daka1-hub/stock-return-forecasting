"""
Clean the raw AAPL news file for downstream use (e.g. sentiment features).

Reads : <Masters>/data_collection/data/collected/tickers/Raw_AAPL_news/nyt_apple_related_news.csv
        (columns: search_term, pub_date, headline, snippet)
Writes: AAPL_news/AAPL_news.csv
        (columns: Date, AAPL_headline_snippet)

Transformations
    1. Drop 'search_term' (not needed downstream).
    2. Rename 'pub_date' -> 'Date', matching the Date column convention used
       throughout the rest of the pipeline.
    3. Merge 'headline' and 'snippet' into one text column,
       'AAPL_headline_snippet' ("headline. snippet"; snippet is missing for
       some rows -- 64 in the raw file -- so those rows keep just the
       headline rather than appending ". nan").
    4. Strip stray literal double-quote characters from the text (e.g.
       quoted speech inside a headline/snippet). Normal CSV quoting is still
       used when writing the file, so commas within the text do not break
       row/column alignment -- only literal " characters inside the text
       itself are removed, the file remains a valid, correctly-parseable CSV.

Run:
    python pre_news.py
"""

from __future__ import annotations
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
COLLECTED_ROOT = REPO_ROOT / "data_collection" / "data" / "collected"

import pandas as pd

SRC_DEFAULT = COLLECTED_ROOT / "tickers" / "Raw_AAPL_news" / "nyt_apple_related_news.csv"
OUT_DEFAULT = HERE / "AAPL_news"


def strip_quotes(x):
    return x.replace('"', "") if isinstance(x, str) else x


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean the raw AAPL news file.")
    ap.add_argument("--src", default=str(SRC_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    src = Path(args.src)
    df = pd.read_csv(src)

    before_rows, before_cols = df.shape

    # 1. drop search_term
    df = df.drop(columns=["search_term"])

    # 2. pub_date -> Date
    df = df.rename(columns={"pub_date": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    # 3. merge headline + snippet
    headline = df["headline"].fillna("").map(strip_quotes).str.strip()
    snippet = df["snippet"].fillna("").map(strip_quotes).str.strip()
    has_snippet = df["snippet"].notna() & (snippet.str.len() > 0)

    # strip the headline's own trailing period(s) before joining, so a
    # headline that already ends in "." (179 of them do) doesn't produce a
    # doubled ".." at the join boundary, e.g. "...F.B.I.. Apple does not..."
    headline_for_merge = headline.str.rstrip(".")
    merged = headline_for_merge.str.cat(snippet, sep=". ").str.strip()
    # rows with no snippet: keep the original headline as-is (its own
    # trailing period, if any, is left untouched)
    merged = merged.mask(~has_snippet, headline)
    df["AAPL_headline_snippet"] = merged

    df = df.drop(columns=["headline", "snippet"])
    df = df[["Date", "AAPL_headline_snippet"]].sort_values("Date")

    dst = Path(args.out) / "AAPL_news.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)

    n_missing = int(df["AAPL_headline_snippet"].isna().sum() +
                    (df["AAPL_headline_snippet"].str.strip() == "").sum())
    print(f"rows: {before_rows} -> {len(df)}   cols: {before_cols} -> {df.shape[1]}")
    print(f"missing headline_snippet: {n_missing}")
    print(f"date range: {df['Date'].min()} .. {df['Date'].max()}")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
