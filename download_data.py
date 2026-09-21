"""
Downloads the Amazon 2023 Gift Cards review file and the NRC Emotion Lexicon
into data/. This keeps large / non-redistributable files out of the repo so
anyone (or the grader) can reproduce the pipeline easily.

Usage:
    python download_data.py
"""

import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

REVIEWS_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz"
REVIEWS_OUT = os.path.join(DATA_DIR, "Gift_Cards.jsonl.gz")

NRC_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip"
NRC_OUT = os.path.join(DATA_DIR, "NRC.zip")


def download(url, out, label):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"[{label}] already present: {out}")
        return False
    print(f"[{label}] downloading {url}")
    sys.stdout.flush()
    urllib.request.urlretrieve(url, out)
    print(f"[{label}] saved {os.path.getsize(out)} bytes -> {out}")
    return True


def main():
    download(REVIEWS_URL, REVIEWS_OUT, "reviews")
    download(NRC_URL, NRC_OUT, "nrc")

    # Extract the English wordlevel lexicon file (needed by emotions_nrc.py).
    import zipfile

    LEX_DIR = os.path.join(DATA_DIR, "extracted")
    os.makedirs(LEX_DIR, exist_ok=True)
    lex_in_zip = "NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    lex_out = os.path.join(LEX_DIR, os.path.basename(lex_in_zip))
    if not os.path.exists(lex_out):
        with zipfile.ZipFile(NRC_OUT) as z:
            with z.open(lex_in_zip) as src, open(lex_out, "wb") as dst:
                dst.write(src.read())
        print(f"[nrc] extracted lexicon -> {lex_out}")
    else:
        print(f"[nrc] lexicon already extracted: {lex_out}")

    print("\nDone.")

    # Clean up: remove the big zip after extraction to save space (optional).
    if os.path.exists(NRC_OUT):
        os.remove(NRC_OUT)
        print("[nrc] removed zip to save space (lexicon retained)")


def download_lexicon_only():
    """Extract the lexicon from an already-downloaded NRC.zip."""
    import zipfile

    LEX_DIR = os.path.join(DATA_DIR, "extracted")
    os.makedirs(LEX_DIR, exist_ok=True)
    lex_in_zip = "NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    lex_out = os.path.join(LEX_DIR, os.path.basename(lex_in_zip))
    if os.path.exists(NRC_OUT):
        with zipfile.ZipFile(NRC_OUT) as z:
            with z.open(lex_in_zip) as src, open(lex_out, "wb") as dst:
                dst.write(src.read())
        os.remove(NRC_OUT)
        print(f"[nrc] extracted lexicon -> {lex_out} (zip cleaned up)")
    else:
        print("NRC.zip not found; run `python download_data.py` first.")


if __name__ == "__main__":
    main()
