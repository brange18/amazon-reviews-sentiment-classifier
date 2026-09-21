"""
MBAX 6418 - Assignment 1
Scoring pipeline.

Loads Gift Card reviews, labels each with the "correct answer" derived from the
star rating (labeling scheme is configurable), runs the LLM classifier on the
title+text, compares the two, and writes results + metrics to files.

The rating is used ONLY to label the answer afterwards; the classifier never
sees it.
"""

import argparse
import gzip
import json
import os
from collections import Counter

import pandas as pd

from classifier import get_client, call_llm, parse_sentiment

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "Gift_Cards.jsonl.gz")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# --- labeling (the "correct answer") ----------------------------------------
def label_2class(rating: float) -> str:
    """Two-class coding from Step 2: >=4 positive, else negative."""
    return "POSITIVE" if rating >= 4 else "NEGATIVE"


def load_reviews(path=DATA_FILE):
    """Stream-load the gzipped JSONL into a DataFrame."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_all(path=DATA_FILE):
    """Cached load of the full review frame."""
    df = load_reviews(path)
    df["rating_int"] = df["rating"].astype(int)
    return df


def corr_yes(ok: bool) -> str:
    return "correct" if ok else "WRONG"


def main():
    ap = argparse.ArgumentParser(description="Score the LLM classifier vs rating")
    ap.add_argument("--rows", type=int, default=100, help="number of rows to score")
    ap.add_argument("--label", choices=["2class", "3class"], default="2class",
                    help="labeling scheme for the correct answer")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--balanced", action="store_true",
                    help="balanced sample per class (uses --per-class)")
    ap.add_argument("--per-class", type=int, default=50,
                    help="rows per class when --balanced")
    ap.add_argument("--with-emotion", action="store_true",
                    help="also ask the LLM for the primary emotion per review")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    df = load_all()
    rng = __import__("random").Random(args.seed)

    if args.balanced:
        # Balanced per-class sample, fixed seed.
        if args.label == "3class":
            df["cls"] = df["rating_int"].apply(lambda r: "POSITIVE" if r >= 4 else ("NEUTRAL" if r == 3 else "NEGATIVE"))
            classes = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
        else:
            df["cls"] = df["rating_int"].apply(label_2class)
            classes = ["POSITIVE", "NEGATIVE"]
        frame = pd.DataFrame()
        for c in classes:
            sub = df[df["cls"] == c]
            k = min(args.per_class, len(sub))
            frame = pd.concat([frame, sub.sample(k, random_state=args.seed)], ignore_index=True)
    else:
        # Sequential first-N rows (Step 2 behavior).
        frame = df.head(args.rows).copy()
        if args.label == "3class":
            frame["cls"] = frame["rating_int"].apply(lambda r: "POSITIVE" if r >= 4 else ("NEUTRAL" if r == 3 else "NEGATIVE"))
        else:
            frame["cls"] = frame["rating_int"].apply(label_2class)

    # Compute correct labels (never passed to the model).
    if args.label == "3class":
        frame["label_answer"] = frame["rating_int"].apply(
            lambda r: "POSITIVE" if r >= 4 else ("NEUTRAL" if r == 3 else "NEGATIVE"))
    else:
        frame["label_answer"] = frame["rating_int"].apply(label_2class)

    client = get_client()
    n_classes = 3 if args.label == "3class" else 2

    # Score each review: llm sees only title+text.
    preds = []
    emos = []
    errors = []
    for idx, row in frame.iterrows():
        title = str(row["title"])
        text = str(row["text"])
        try:
            raw = call_llm(client, title, text, n_classes=n_classes,
                           with_emotion=args.with_emotion)
            parsed = parse_sentiment(raw, with_emotion=args.with_emotion)
            preds.append(parsed["sentiment"])
            emos.append(parsed.get("emotion"))
        except Exception as exc:  # noqa: BLE001
            preds.append(None)
            emos.append(None)
            errors.append({"index": idx, "title": title, "error": str(exc)[:300]})
        if len(preds) % 10 == 0:
            print(f"  scored {len(preds)}/{len(frame)}")

    frame["pred"] = preds
    frame["emotion"] = emos
    frame["correct"] = [
        None if p is None else (p == a)
        for p, a in zip(frame["pred"], frame["label_answer"])
    ]

    # Aggregate metrics.
    scored = frame[frame["pred"].notna()].copy()
    n_total = len(frame)
    n_confident = len(scored)
    acc = float(scored["correct"].mean()) if n_confident else 0.0
    metrics = {
        "label_scheme": args.label,
        "n_rows_attempted": n_total,
        "n_classifications": n_confident,
        "n_errors": len(errors),
        "accuracy": round(acc, 4),
        "accuracy_pct": round(100 * acc, 2),
        "per_class": {},
    }
    for c in sorted(frame["label_answer"].unique()):
        sub = scored[scored["label_answer"] == c]
        metrics["per_class"][c] = {
            "n_in_batch": int((frame["label_answer"] == c).sum()),
            "n_scored": int(len(sub)),
            "correct": int((sub["correct"] == True).sum()),  # noqa: E712
            "accuracy": round(float(sub["correct"].mean()), 4) if len(sub) else None,
        }

    # Confusion matrix (answer rows x predicted cols).
    classes_all = sorted(scored["label_answer"].unique().tolist()) or []
    for p in scored["pred"].unique():
        if p not in classes_all:
            classes_all.append(p)
    classes_all = sorted(set(classes_all), key=lambda x: (x != "POSITIVE", x))
    conf = {}
    for a in classes_all:
        for p in classes_all:
            conf[f"{a}->{p}"] = int(((scored["label_answer"] == a) & (scored["pred"] == p)).sum())

    print("\n===== RESULTS =====")
    print(f"label scheme        : {args.label}")
    print(f"reviews attempted   : {n_total}")
    print(f"classifications     : {n_confident}")
    print(f"errors              : {len(errors)}")
    print(f"accuracy            : {metrics['accuracy_pct']:.1f}%")
    print("\nper-class accuracy:")
    for c, m in metrics["per_class"].items():
        a = f"{m['accuracy']*100:.1f}%" if m["accuracy"] is not None else "n/a"
        print(f"  {c:<9} n={m['n_scored']:>4}  correct={m['correct']:>4}  acc={a}")
    print("\nconfusion (answer -> prediction):")
    for k, v in conf.items():
        if v:
            print(f"  {k:<20} {v}")

    if errors:
        print(f"\n!! {len(errors)} classification errors (see saved output)")

    # Save outputs.
    if not args.no_save:
        os.makedirs(OUT_DIR, exist_ok=True)
        tag = args.tag or args.label
        out_rows = frame[["rating", "rating_int", "title", "text", "label_answer", "pred", "emotion", "correct"]].copy()
        out_csv = os.path.join(OUT_DIR, f"results_{tag}.csv")
        out_rows.to_csv(out_csv, index=False)
        pred_json = os.path.join(OUT_DIR, f"predictions_{tag}.jsonl")
        with open(pred_json, "w", encoding="utf-8") as f:
            for _, r in out_rows.iterrows():
                f.write(json.dumps({
                    "rating": r["rating"], "rating_int": r["rating_int"],
                    "title": r["title"], "text": r["text"],
                    "answer": r["label_answer"], "predicted": r["pred"],
                    "emotion": r["emotion"], "correct": r["correct"],
                }, ensure_ascii=False) + "\n")
        summary = {
            "tag": tag, "params": vars(args), "metrics": metrics, "confusion": conf,
            "errors": errors,
        }
        with open(os.path.join(OUT_DIR, f"summary_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nsaved -> {OUT_DIR}/results_{tag}.csv, predictions_{tag}.jsonl, summary_{tag}.json")


if __name__ == "__main__":
    main()
