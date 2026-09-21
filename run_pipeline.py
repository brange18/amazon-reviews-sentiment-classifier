"""
MBAX 6418 - Assignment 1
End-to-end pipeline helper.

Takes a raw predictions jsonl (from scoring.py), adds the NRC word-list emotion
columns, feeds it to the dashboard generator, and prints the dashboard file.
Also prints a short human-readable summary.

Usage:
    python run_pipeline.py results/predictions_balanced3.jsonl -o dashboard.html
"""

import argparse
import json
import os
import sys

from build_dashboard import build_html, build_metrics, read_predictions
from emotions_nrc import add_emotion_columns, load_lexicon


def write_with_nrc(pred_jsonl, out_jsonl):
    lexicon = load_lexicon()
    rows = read_predictions(pred_jsonl)
    add_emotion_columns(rows)
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows, lexicon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_jsonl", help="predictions jsonl from scoring.py")
    ap.add_argument("-o", "--out", default="dashboard.html")
    ap.add_argument("--title", default="Amazon Gift Card Review · Sentiment & Emotion")
    ap.add_argument("--subtitle",
                    default="LLM sentiment + emotion prediction checked against the "
                            "star-rating answer, plus an NRC word-list emotion baseline.")
    ap.add_argument("--keep", action="store_true",
                    help="keep the intermediate *_nrc.jsonl file")
    args = ap.parse_args()

    base = os.path.splitext(args.pred_jsonl)[0]
    nrc_jsonl = base + "_nrc.jsonl"

    print("Loading lexicon + adding NRC emotions...", file=sys.stderr)
    rows, lexicon = write_with_nrc(args.pred_jsonl, nrc_jsonl)
    print(f"  lex words: {len(lexicon)}", file=sys.stderr)

    metrics = build_metrics(rows)
    doc = build_html({
        "rows": rows, "metrics": metrics,
        "title": args.title, "subtitle": args.subtitle,
    })
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)

    print(f"dashboard: {args.out} ({os.path.getsize(args.out):,} bytes)")
    if not args.keep and nrc_jsonl != args.out:
        try:
            os.remove(nrc_jsonl)
        except OSError:
            pass

    # Quick textual recap.
    n = len(rows)
    n_pred = sum(1 for r in rows if r.get("predicted"))
    acc = 100 * metrics["acc"]
    print(f"\nrecap: {n_pred}/{n} classified · accuracy {acc:.2f}%")
    nrc_has = sum(1 for r in rows if r.get("nrc_emotion"))
    llm_has = sum(1 for r in rows if r.get("emotion"))
    both = [r for r in rows if r.get("nrc_emotion") and r.get("emotion")]
    agree = 100 * sum(1 for r in both if r["nrc_emotion"] == r["emotion"]) / len(both) if both else None
    print(f"NRC emotion present on {nrc_has}/{n} rows")
    print(f"LLM emotion present on {llm_has}/{n} rows")
    if agree is not None:
        print(f"LLM vs NRC emotion agreement: {agree:.1f}% over {len(both)} rows with both")


if __name__ == "__main__":
    main()
