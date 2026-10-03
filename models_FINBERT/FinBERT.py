"""
Run FinBERT (ProsusAI/finbert) over each line of the cleaned AAPL news file
to produce a continuous sentiment score in [-1, 1]:
    -1  most negative for the stock
     0  neutral
    +1  most positive for the stock
(and anything in between).

Score = P(positive) - P(negative), taken from FinBERT's 3-class softmax
output (positive / negative / neutral). This is bounded in [-1, 1] by
construction -- both probabilities are >= 0 and the three classes sum to 1
-- and naturally collapses toward 0 when the model is confidently neutral,
since P(positive) and P(negative) are then both small.

Reads : <Masters>/data_preprocess/feature_engineering/AAPL_news/AAPL_news.csv
        (columns: Date, AAPL_headline_snippet)
Writes: <Masters>/data_preprocess/feature_engineering/AAPL_news/AAPL_scores.csv
        (columns: Date, AAPL_sentiment_score) -- one row per input line.

Requires: torch, transformers.
    This machine's torch_cuda conda env has both already:
        conda run -n torch_cuda python FinBERT.py

Run:
    python FinBERT.py
    python FinBERT.py --src ... --out ... --batch-size 32
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "ProsusAI/finbert"

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC_DEFAULT = REPO_ROOT / "data_preprocess" / "feature_engineering" / "AAPL_news" / "AAPL_news.csv"
OUT_DEFAULT = REPO_ROOT / "data_preprocess" / "feature_engineering" / "AAPL_news" / "AAPL_scores.csv"


def score_texts(texts: list[str], tokenizer, model, device: str,
                batch_size: int) -> list[float]:
    id2label = {i: l.lower() for i, l in model.config.id2label.items()}
    pos_idx = next(i for i, l in id2label.items() if l == "positive")
    neg_idx = next(i for i, l in id2label.items() if l == "negative")

    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(device)
            probs = torch.softmax(model(**inputs).logits, dim=-1)
            scores.extend((probs[:, pos_idx] - probs[:, neg_idx]).tolist())
            print(f"  scored {min(start + batch_size, len(texts))}/{len(texts)}",
                  flush=True)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description="FinBERT sentiment scores for AAPL news.")
    ap.add_argument("--src", default=str(SRC_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    src = Path(args.src)
    df = pd.read_csv(src)
    texts = df["AAPL_headline_snippet"].astype(str).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

    scores = score_texts(texts, tokenizer, model, device, args.batch_size)

    out_df = pd.DataFrame({"Date": df["Date"], "AAPL_sentiment_score": scores})

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(dst, index=False)

    print(f"\nrows scored: {len(out_df)}")
    print(f"score range: [{out_df['AAPL_sentiment_score'].min():.4f}, "
          f"{out_df['AAPL_sentiment_score'].max():.4f}]")
    print(f"mean: {out_df['AAPL_sentiment_score'].mean():.4f}")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
