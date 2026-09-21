"""
MBAX 6418 - Assignment 1
NRC word-list emotion scorer (Step 5b).

Scores each review's words against the NRC Word-Emotion Association Lexicon
(EmoLex), sums the association count per emotion, and takes the highest-scoring
emotion as the word-list-derived primary emotion.

Emotions used (from the lexicon): anger, anticipation, disgust, fear, joy,
sadness, surprise, trust.

No model calls -- runs purely over text already in the predictions files.
"""

import json
import os
import re

# Where the (downloaded, not committed) lexicon lives.
LEX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "extracted", "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
)

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]

# If the model ends up selecting none (tie/negative score), a fallback emotion.
DEFAULT_EMOTION = "joy"


def load_lexicon(path=LEX_PATH):
    """Load the NRC wordlevel file into {word: set(emotions)}.

    File format (tab separated, Windows CRLF):
        word<TAB>emotion<TAB>0|1
    Only entries with a 1 keep the emotion for that word.
    """
    lex = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            if len(parts) < 3:
                continue
            word, emotion, flag = parts[0], parts[1], parts[2]
            if flag.strip() != "1":
                continue
            lex.setdefault(word, set()).add(emotion)
    return lex


def score_text(text: str, lexicon: dict):
    """Return {emotion: int} association counts for a piece of text."""
    scores = {e: 0 for e in EMOTIONS}
    # Lowercase, strip punctuation, tokenize on non-alpha.
    tokens = re.findall(r"[a-z']+", (text or "").lower())
    for tok in tokens:
        emo = lexicon.get(tok)
        if emo:
            # Only the 8 emotion categories count; positive/negative are
            # sentiment tags in the lexicon and are excluded from "emotion".
            for e in emo:
                if e in scores:
                    scores[e] += 1
    return scores


def primary_emotion(scores: dict):
    """Return the emotion with the highest raw score (ties -> first in order)."""
    best = max(EMOTIONS, key=lambda e: scores[e])
    if scores[best] == 0:
        return None
    return best


def add_emotion_columns(pred_rows):
    """Given loaded prediction rows (dicts), add nrc emotion scores + label."""
    lexicon = load_lexicon()
    for r in pred_rows:
        text = f"{r.get('title', '')} {r.get('text', '')}"
        scores = score_text(text, lexicon)
        r["nrc_scores"] = scores
        r["nrc_emotion"] = primary_emotion(scores)


def process_file(pred_jsonl: str):
    """Load a predictions jsonl, add NRC emotion columns, write back/return."""
    with open(pred_jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    add_emotion_columns(rows)
    out = os.path.splitext(pred_jsonl)[0] + "_nrc.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out, rows


if __name__ == "__main__":
    import sys

    lex = load_lexicon()
    print(f"lexicon loaded: {len(lex)} words, {sum(len(v) for v in lex.values())} word-emotion associations")

    demo = [
        "I absolutely love this gift card, so happy and thrilled!",
        "This is the worst scam ever, angry and disgusted by the company.",
        "I fear the card will not work, disappointed and sad about it.",
    ]
    for t in demo:
        s = score_text(t, lex)
        print("----")
        print(t)
        print("scores:", {k: v for k, v in s.items() if v})
        print("primary:", primary_emotion(s))

    if len(sys.argv) > 1:
        out, rows = process_file(sys.argv[1])
        print(f"\nwrote {out} with {len(rows)} rescored rows")
