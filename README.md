# MBAX 6418 · Assignment 1 — Sentiment & Emotion Classification of Amazon Reviews

A working review-sentiment classifier for **Amazon Gift Card** reviews. It
classifies each review as **POSITIVE / NEUTRAL / NEGATIVE**, detects the
**primary emotion** two independent ways (an LLM prediction and an NRC word-list
derivation), checks the predictions against the **star-rating answer**, and
presents everything in a single self-contained, offline HTML dashboard.

**Data source:** [Amazon Reviews '23](https://amazon-reviews-2023.github.io) —
a large-scale product-review dataset collected by the **McAuley Lab at UC San
Diego**. We use the **Gift Cards** review category
(`review_categories/Gift_Cards.jsonl.gz`), a gzipped JSON-Lines file of
**152,410 individual reviews**. Each line is one review with `rating`, `title`,
`text`, and metadata (`verified_purchase`, `helpful_vote`, `timestamp`, `images`,
`asin`, `parent_asin`, `user_id`).

---

## Highlights

| Metric | Value |
|---|---|
| Gift Card reviews in the file | 152,410 |
| Star distribution (whole file) | 5★ 84.1% · 4★ 4.4% · 3★ 2.1% · 2★ 1.2% · 1★ 8.1% — heavily 5★-skewed |
| Sequential 100-row, 2-class accuracy | **98.0%** (98/100) |
| Balanced 3-class accuracy (seed 42, 50/class) | **73.3%** (110/150) |
| LLM vs. NRC primary-emotion agreement | **19.2%** (over 120 rows with both) |

---

## Why the lopsided run looked too good

The file is dominated by ★★★★★ reviews — mapped to sentiment, roughly **88% of
the whole file is positive**. A classifier that usually guesses "POSITIVE" will
be right most of the time without actually learning to separate the classes.

On the **sequential first-100** batch (98.0% accuracy) this is exactly what
happened: the batch happened to contain **93 positive and only 7 negative**
reviews. The model aced the easy positive majority (92/93) and handled the tiny
negative minority (6/7), which flatters the headline number. That result tells
you the model is *decent*, but nothing about how it handles rare classes —
there simply weren't enough negative (and no neutral) reviews in the batch.

To see the real picture we re-scored a **balanced** sample: **50 reviews drawn
from each of the three classes** with a **fixed random seed (42)** so the exact
same set comes up every time. With equal class representation the headline
accuracy falls and per-class differences become visible.

---

## Model, prompt, and endpoint

- **Model:** `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` served on the class
  OpenAI-compatible endpoint (`http://dobolyi.com:9001/v1`).
- **Input to model:** only `title` + `text`. The star rating is used **solely**
  to label the answer afterwards; it never reaches the model.
- **Structured prompt:** defined in [`classifier.py`](classifier.py). It asks for
  a strict single JSON object per review, so results are read back
  programmatically, with tolerant parsing for code fences / trailing prose.
- Prompts exist for the 2-class (POSITIVE/NEGATIVE, Steps 1–2) and 3-class
  (POSITIVE/NEUTRAL/NEGATIVE, Step 6) tasks, and optionally add primary-emotion
  detection (Step 5) in the same call.

### Spot check (Step 1)
Before scoring, the prompt was run on hand-written obvious and edge cases and
got them all right — including a **positive title / negative body** review
(*"Great looking card — but it would not load, very frustrating"* → NEGATIVE)
and a **terse** one (*"Ugh … so disappointed"* → NEGATIVE).

---

## How it's built (files)

| File | Purpose |
|---|---|
| `classifier.py` | The prompt + OpenAI-compatible LLM plumbing (call, disk cache, parsing). |
| `scoring.py` | Loads reviews, labels the answer from the rating, runs the classifier, saves results + a JSON summary. Supports sequential and balanced sampling + emotion. |
| `emotions_nrc.py` | Loads the NRC Word-Emotion Association Lexicon and derives a per-review primary emotion purely from word counts (no model calls). |
| `build_dashboard.py` | Generates the self-contained HTML dashboard from a predictions file. |
| `run_pipeline.py` | One-call convenience: add NRC columns → build dashboard → print recap. |
| `download_data.py` | Downloads the reviews file + NRC lexicon (keeps large / non-redistributable data out of the repo). |
| `results/` | Saved raw predictions (`predictions_*.jsonl`), summaries (`summary_*.json`), and dashboards. |
| `shots/` | Screenshots of the dashboard used in this report. |

**Reproduce:**
```bash
pip install -r requirements.txt
python download_data.py            # fetches data/ (reviews + NRC lexicon), gitignored
python scoring.py --balanced --per-class 50 --label 3class --with-emotion --seed 42   # Step 6 run
python run_pipeline.py results/predictions_balanced3.jsonl -o dashboard.html           # NRC + dashboard
```

> The NRC Emotion Lexicon is © National Research Council Canada and is licensed
> for non-commercial research/educational use only (no redistribution). This
> repository therefore does **not** commit it; `download_data.py` fetches it on
> demand and `.gitignore` keeps it out of version control.

---

## Results

### Step 2 — Sequential 100-row, 2-class
- **98.0%** accuracy (98/100), 0 errors.
- Shared verdict: POSITIVE **92/93 (98.9%)**, NEGATIVE **6/7 (85.7%)**.
- The 2 disagreements: 1 positive review called negative, 1 negative review
  called positive. Saved in `results/predictions_seq100.jsonl`.

### Step 6 — Balanced three-class (seed 42, 50 per class)
- **73.3%** accuracy (110/150), 0 errors. Every number below is read directly
  from `results/summary_balanced3.json`.
- Per-class accuracy:
  - POSITIVE: **48/50 = 96.0%**
  - NEUTRAL: **14/50 = 28.0%**
  - NEGATIVE: **48/50 = 96.0%**
- **The ★★★ neutral class collapses into NEGATIVE.** Of the 50 true 3★ reviews:
  only **14** were called NEUTRAL, while **31 were dragged into NEGATIVE** (62%)
  and 5 into POSITIVE. In other words, the model treats a "lukewarm / mixed"
  review as a complaint. Negative also bleeds slightly into neutral (2 rows), and
  positive barely leaks (1 → negative, 1 → neutral).

> **What the balanced run revealed:** the lopsided 98% hid both the model's real
> strengths (nearly perfect on clearly positive/negative text) and its one true
> weak spot (neutral ★★★ reviews). Neutral is where almost all the mistakes
> concentrate — the model would rather call an ambiguous review *negative* than
> *neutral*.

### Step 5 — Primary emotion: LLM vs. NRC word list
- The **LLM** predicts one of 8 emotions (anger, anticipation, disgust, fear,
  joy, sadness, surprise, trust) alongside sentiment in the same call. It
  produced an emotion on **150/150** rows.
- The **NRC word list** is matched against each review's words; emotion counts
  are summed and the highest is the answer. It produced an emotion on
  **120/150** rows (30 reviews had no NRC-listed emotion word). No model calls.
- Agreement: **19.2%** over the **120 rows with both**.
- Non-overlapping trends: the LLM favors **anger (66)** and **joy (53)**,
  whereas the word list favors **anticipation (68)** and **joy (20)**. Two real
  reasons they diverge:
  1. **NRC weights function words and neutral double-meanings** — e.g. *"It's a
     gift card"* matches *expect*/"card" adjectives that score high on
     *anticipation*, inflating it far above the review's vibe.
  2. **"disappointed" is tagged as anger in NRC**, so a neutral-but-leery review
     like *"Gift message not included. Very disappointing"* becomes **anger** by
     word list but **sadness** by the LLM — a lexicon quirk, not shared meaning.

---

## Built with an agent (process & bugs found)

- Drove every step with the project agent: chose the OpenAI-compatible SDK, the
  structured-JSON prompt, the fixed-seed balanced sample, and a single-file
  offline dashboard.
- **Interactive-table bug:** the review table initially reported *"Showing 0"* —
  the JS keyed rows by an index missing from the serialized data. Fixed by
  embedding the row index and re-verified in the browser.
- **Chart bugs:** a `max(dist.values(), 1)` normalization call raised
  `TypeError` (mixing `dict_values` with `int`), and the "total count" was a
  static placeholder that never updated. Both fixed; the filter now shows a live
  count and was clicked-through in the preview ("Wrong only" → exactly the 2
  mismatched rows).
- **NRC quirk:** near-verbatim neutral titles ("It's a gift card") can dominate
  the word-list score, pushing anticipation/trust high even on terse reviews.

## Dashboard

Open **`dashboard.html`** (or `results/dashboard_balanced3.html`) in any
browser — data, styling, and interactivity are all in one file; no server or
network needed. It shows headline KPIs, the star-rating distribution, correct
answer vs. predicted, per-class accuracy, a confusion matrix, the LLM vs. NRC
emotion comparison, and a **filterable review table** (all / correct-only /
wrong-only / unscored, by true class, by predicted class, free-text search) with
a live count.

![Dashboard top](./shots/dashboard_full.png)

---

## Author / review

This project and report were generated with the help of the assignment's agent
workflow; every number quoted is read directly from the saved output under
`results/` (see `summary_seq100.json` and `summary_balanced3.json`) and was
re-checked against the rendered dashboard. Final review and verification was
performed by the student submitting it.
