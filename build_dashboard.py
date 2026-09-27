"""
MBAX 6418 - Assignment 1
Dashboard generator.

Reads one (or several) saved prediction/result files and emits a single
self-contained, offline HTML page: headline metrics, star distribution,
class accuracy, confusion, emotion views, and a filterable review table.

Usage:
    python build_dashboard.py --predictions results/predictions_balanced3.jsonl
                              --out dashboard.html
"""

import argparse
import base64
import html
import json
import os
from collections import Counter


def human_int(n):
    s = f"{n:,}"
    return s


def esc(text):
    return html.escape(str(text) if text is not None else "")


def read_predictions(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def class_of(rating):
    r = int(float(rating))
    if r >= 4:
        return "POSITIVE"
    if r == 3:
        return "NEUTRAL"
    return "NEGATIVE"


def build_metrics(rows):
    n = len(rows)
    n_pred = sum(1 for r in rows if r.get("predicted"))
    acc = sum(1 for r in rows if r.get("correct")) / n if n else 0
    answer_dist = Counter(class_of(r["rating"]) for r in rows)
    pred_dist = Counter(r.get("predicted") for r in rows if r.get("predicted"))
    return {
        "n": n, "n_pred": n_pred, "acc": round(acc, 4),
        "acc_pct": round(100 * acc, 1),
        "answer_dist": dict(answer_dist), "pred_dist": dict(pred_dist),
    }


def insert_emoji(emotion):
    return {
        "anger": "😠", "anticipation": "🔮", "disgust": "🤢", "fear": "😨",
        "joy": "😄", "sadness": "😢", "surprise": "😲", "trust": "🤝",
        None: "—",
    }.get(emotion, "—")


CLASS_COLOR = {"POSITIVE": "#2e7d32", "NEUTRAL": "#e6a700", "NEGATIVE": "#c62828"}


def build_html(data):
    rows = data["rows"]
    m = data["metrics"]

    # ---- aggregates for charts ----
    classes = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    answer_dist = m["answer_dist"]
    pred_dist = m["pred_dist"]

    # per-class accuracy
    per_class_acc = {}
    per_class_counts = {}
    for c in classes:
        sub = [r for r in rows if class_of(r["rating"]) == c]
        n_c = len(sub)
        per_class_counts[c] = n_c
        ok = sum(1 for r in sub if r.get("correct"))
        per_class_acc[c] = (round(100 * ok / n_c, 1) if n_c else 0)

    # confusion (answer -> predicted) -- full grid, zeros shown
    conf = {}
    all_labels = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    for a in all_labels:
        for p in all_labels:
            conf[f"{a}|{p}"] = sum(
                1 for r in rows if class_of(r["rating"]) == a and r.get("predicted") == p)

    # star distribution
    star_dist = dict(Counter(int(float(r["rating"])) for r in rows))
    stars_sorted = sorted(star_dist.items())
    star_max = max(star_dist.values()) or 1

    # emotion comparison (if present)
    nrc_emo = None
    llm_emo = None
    emo_agree = None
    emo_agree_all = None
    n_nrc_missing = None
    if rows and "nrc_emotion" in rows[0]:
        nrc_emo = Counter(r.get("nrc_emotion") for r in rows)
        llm_emo = Counter(r.get("emotion") for r in rows)
        both = [r for r in rows if r.get("nrc_emotion") and r.get("emotion")]
        emo_agree = (round(100 * sum(1 for r in both if r["nrc_emotion"] == r["emotion"]) / len(both), 1)
                     if both else None)
        # agreement over ALL rows (a row with a missing word-list emotion counts
        # as "no agreement"), plus how many rows the word list left unlabeled.
        n_all = len(rows)
        agree_all = sum(1 for r in rows if r.get("nrc_emotion") and r.get("emotion")
                        and r["nrc_emotion"] == r["emotion"])
        emo_agree_all = round(100 * agree_all / n_all, 1) if n_all else None
        n_nrc_missing = sum(1 for r in rows if not r.get("nrc_emotion"))

    # runs comparison (optional, from run_pipeline)
    runs = data.get("runs") or []
    runs_html = ""
    if runs:
        rows_html = ""
        for rn in runs:
            rows_html += (
                f'<tr><td>{esc(rn.get("name", ""))}</td>'
                f'<td>{esc(str(rn.get("scheme", "")))}</td>'
                f'<td>{rn.get("n", "&ndash;")}</td>'
                f'<td class="num">{rn.get("acc", "&ndash;")}</td></tr>'
            )
        runs_html = f"""
        <section class="card">
          <h2 class="sec-title">🔁 All scored runs</h2>
          <p class="sub">The balanced sample is the main run. The sequential first-100 is included for comparison: its easy, 5★-heavy mix inflates the headline number.</p>
          <table><thead><tr><th>Run</th><th>Scheme</th><th>Reviews</th><th>Accuracy</th></tr></thead>
          <tbody>{rows_html}</tbody></table>
        </section>"""

    # whole-file distribution (optional, from run_pipeline)
    whole = data.get("whole_file")
    whole_bars = ""
    whole_panel = ""
    if whole:
        wstars = whole.get("stars", {})
        wtotal = whole.get("total", 0)
        wmax = max(wstars.values()) or 1
        for s in range(1, 6):
            v = wstars.get(s, 0)
            whole_bars += (
                f'<div class="bar-row"><span class="bar-label">{s}★</span>'
                f'<div class="bar-track"><div class="bar-fill st{s}" '
                f'style="width:{100*v/wmax:.2f}%"></div></div>'
                f'<span class="bar-val">{v:,} · {100*v/wtotal:.1f}%</span></div>'
            )
        whole_panel = (
            f'<div class="panel">{whole_bars}'
            f'<div class="panel-title">Whole file · {wtotal:,} reviews</div></div>'
        )
    else:
        whole_panel = "<p class='sub'>whole-file distribution not provided</p>"

    # ---- assemble JS data ----
    js_rows = []
    for idx, r in enumerate(rows):
        js_rows.append({
            "_i": idx,
            "rating": r["rating"], "title": r["title"], "text": r["text"],
            "answer": class_of(r["rating"]), "predicted": r.get("predicted"),
            "correct": r.get("correct"),
            "emotion": r.get("emotion"), "nrc_emotion": r.get("nrc_emotion"),
        })
    js_data = json.dumps(js_rows)

    star_bars = "".join(
        f'<div class="bar-row"><span class="bar-label">{s}★</span>'
        f'<div class="bar-track"><div class="bar-fill st{s}" style="width:{100*v/star_max:.1f}%"></div></div>'
        f'<span class="bar-val">{v:,}</span></div>'
        for s, v in stars_sorted)

    # full 3x3 confusion grid (zeros are a finding too)
    def conf_cell(a, p):
        n = conf.get(f"{a}|{p}", 0)
        diag = "diag" if a == p else ""
        zero = "zero" if n == 0 else ""
        return (f'<td class="conf-cell {diag} {zero}" data-a="{esc(a)}" data-p="{esc(p)}">'
                f'{n}</td>')

    conf_rows = "".join(
        f'<tr class="conf-row" data-a="{esc(a)}">'
        f'<td class="cell-a">{esc(a)}</td>'
        + "".join(conf_cell(a, p2) for p2 in all_labels)
        + f'</tr>'
        for a in all_labels
    )
    conf_head = "<tr><th>true \\ pred</th>" + "".join(f"<th>{esc(p)}</th>" for p in all_labels) + "</tr>"

    # per-class accuracy bars
    acc_bars = ""
    for c in classes:
        col = CLASS_COLOR[c]
        v = per_class_acc[c]
        acc_bars += (
            f'<div class="bar-row"><span class="bar-label"><span class="dot" '
            f'style="background:{col}"></span>{c}<span class="bar-sub">n={per_class_counts[c]:,}</span></span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{v:.1f}%;background:{col}"></div></div>'
            f'<span class="bar-val">{v:.1f}%</span></div>'
        )

    # answer vs predicted distribution side by side
    def dist_chart(name, dist, cls):
        total = sum(dist.values()) or 1
        bars = ""
        for c in cls:
            v = dist.get(c, 0)
            pct = 100 * v / total
            bars += (
                f'<div class="bar-row"><span class="bar-label">{c}</span>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{CLASS_COLOR[c]}"></div></div>'
                f'<span class="bar-val">{v:,} · {pct:.1f}%</span></div>'
            )
        return f'<h4>{name}</h4>{bars}'

    answer_chart = dist_chart("Correct answer (from rating)", answer_dist, classes)
    pred_chart = dist_chart("Model prediction", pred_dist, classes)

    # emotion panels
    emo_panel = ""
    if nrc_emo is not None:
        def emo_bars(dist):
            total = sum(dist.values()) or 1
            out = ""
            order = ["joy", "trust", "anticipation", "surprise", "fear", "sadness", "anger", "disgust"]
            for e in order:
                v = dist.get(e, 0)
                out += (f'<div class="bar-row"><span class="bar-label">{insert_emoji(e)} {esc(e)}</span>'
                        f'<div class="bar-track"><div class="bar-fill emo" style="width:{100*v/total:.1f}%"></div></div>'
                        f'<span class="bar-val">{v:,}</span></div>')
            return out

        if emo_agree_all is not None:
            agree_line = (
                f"<p class='agree'>Emotion agreement: <b>{emo_agree_all}%</b> across "
                f"all {n_all} reviews ({emo_agree}% over the {len(both)} rows where both "
                f"methods produced an emotion). {n_nrc_missing} rows had no word in the "
                f"NRC list, so they score as no-agreement.</p>"
            )
        else:
            agree_line = ""
        emo_panel = f"""
        <section class="card emo-section">
          <h2 class="sec-title">🕵️ Primary emotion — LLM vs. word list</h2>
          <p class="sub">The LLM predicts one of 8 emotions per review (same call as sentiment). The NRC word list scores each review's words and takes the highest emotion; ties are broken alphabetically (anticipation, disgust, fear…) — which is why anticipation is common in the word-list column.</p>
          {agree_line}
          <div class="grid2">
            <div class="panel">{emo_bars(llm_emo if llm_emo else {})}<div class="panel-title">LLM-predicted emotion ({sum((llm_emo or {}).values()):,} rows)</div></div>
            <div class="panel">{emo_bars(nrc_emo)}<div class="panel-title">NRC word-list emotion ({sum((nrc_emo or {}).values()):,} rows + {n_nrc_missing} no-emotion)</div></div>
          </div>
        </section>"""

    # ---- review table rows ----
    def table_row(idx, r):
        status = ("✓ correct" if r.get("correct") else ("✗ wrong" if r.get("correct") is False else "—"))
        status_class = "ok" if r.get("correct") else ("bad" if r.get("correct") is False else "na")
        a = class_of(r["rating"])
        p = r.get("predicted") or "—"
        return (
            f'<tr id="row-{idx}" '
            f'data-filter-answer="{esc(a)}" data-filter-pred="{esc(p)}" '
            f'data-filter-correct="{("y" if r.get("correct") else "n") if r.get("correct") is not None else ""}">'
            f'<td class="num">{int(float(r["rating"]))}★</td>'
            f'<td class="txt"><span class="t">{esc(r["title"])}</span>'
            f'<span class="rev-body">{esc(r["text"])}</span></td>'
            f'<td>{esc(a)}</td>'
            f'<td>{esc(p)}</td>'
            f'<td class="emo">{insert_emoji(r.get("emotion"))}</td>'
            f'<td class="emo">{insert_emoji(r.get("nrc_emotion"))}</td>'
            f'<td><span class="status {status_class}">{status}</span></td>'
            f'</tr>'
        )

    table_body = "".join(table_row(i, r) for i, r in enumerate(rows))

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{data["title"]}</title>
<style>
:root {{
  --bg:#0f1419; --card:#171e26; --card2:#1d2630; --fg:#e7edf3; --muted:#9aa7b4;
  --line:#2a3540; --accent:#4ea1ff; --ok:#2ecc71; --bad:#e74c3c; --na:#95a5a6;
  --shadow:0 6px 24px rgba(0,0,0,.35);
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font-family:'Segoe UI',system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif; line-height:1.5; }}
.wrap {{ max-width:1120px; margin:0 auto; padding:32px 20px 80px; }}
header.hero {{ padding:28px 0 8px; }}
.overline {{ text-transform:uppercase; letter-spacing:.14em; font-size:12px; color:var(--accent); font-weight:600; }}
h1 {{ margin:6px 0 4px; font-size:30px; font-weight:700; letter-spacing:-.01em; }}
.hero p {{ margin:4px 0; color:var(--muted); font-size:14px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:26px 0; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px 16px; box-shadow:var(--shadow); }}
.kpi .v {{ font-size:30px; font-weight:700; }}
.kpi .l {{ color:var(--muted); font-size:12.5px; margin-top:4px; text-transform:uppercase; letter-spacing:.06em; }}
.kpi.accent .v {{ color:var(--accent); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:22px; margin:18px 0; box-shadow:var(--shadow); }}
.sec-title {{ margin:0 0 6px; font-size:19px; font-weight:600; }}
.sub {{ color:var(--muted); font-size:13.5px; margin:0 0 18px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
@media (max-width:760px){{ .grid2{{grid-template-columns:1fr;}} }}
.panel {{ background:var(--card2); border-radius:12px; padding:14px 16px; }}
.panel-title {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; margin-top:10px; }}
.bar-row {{ display:flex; align-items:center; gap:10px; margin:7px 0; }}
.bar-label {{ flex:0 0 130px; font-size:13px; }}
.bar-sub {{ color:var(--muted); font-size:11px; margin-left:6px; }}
.bar-track {{ flex:1; height:18px; background:#0b0f14; border-radius:9px; overflow:hidden; }}
.bar-fill {{ height:100%; background:var(--accent); border-radius:9px; }}
.bar-fill.st1,.bar-fill.st2{{background:#c62828;}} .bar-fill.st3{{background:#e6a700;}} .bar-fill.st4{{background:#81c784;}} .bar-fill.st5{{background:#2e7d32;}}
.bar-val {{ flex:0 0 auto; font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
table {{ width:100%; border-collapse:collapse; }}
th {{ text-align:left; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; padding:10px 10px; border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--card); }}
td {{ padding:10px; border-bottom:1px solid var(--line); vertical-align:top; font-size:13.5px; }}
td.num {{ white-space:nowrap; font-weight:600; }}
td.txt .t {{ font-weight:600; display:block; }}
td.txt .rev-body {{ color:var(--muted); font-size:12.5px; display:block; margin-top:2px; max-width:480px; }}
td.emo {{ text-align:center; }}
.status {{ font-size:11px; padding:2px 8px; border-radius:20px; font-weight:600; white-space:nowrap; }}
.status.ok{{background:#173d2a;color:var(--ok);}} .status.bad{{background:#47201f;color:var(--bad);}} .status.na{{background:#2a323b;color:var(--na);}}
.pill {{ display:inline-block; padding:1px 10px; border-radius:20px; color:#fff; font-weight:700; font-size:12px; }}
table.conf {{ width:auto; min-width:340px; }}
table.conf th {{ text-align:center; font-size:12px; }}
table.conf td.conf-cell {{ text-align:center; font-weight:700; font-size:15px; padding:10px 18px; border:1px solid var(--line); background:var(--card2); border-radius:8px; }}
table.conf tr {{ background:transparent !important; }}
table.conf td.cell-a {{ font-weight:600; padding:10px 14px; }}
table.conf .diag {{ border-color:var(--accent); color:var(--accent); }}
table.conf .zero {{ color:#4b5563; font-weight:400; }}
.conf-row td.cell-a{{font-weight:600;}} 
.filters {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:16px 0; }}
.filters select,.filters input {{ background:var(--card2); color:var(--fg); border:1px solid var(--line); border-radius:9px; padding:8px 12px; font-size:13.5px; }}
.filters label {{ color:var(--muted); font-size:13px; }}
.chips {{ display:flex; gap:10px; flex-wrap:wrap; }}
.chip {{ padding:8px 14px; border:1px solid var(--line); border-radius:20px; cursor:pointer; font-size:13px; background:var(--card2); color:var(--fg); }}
.chip.on {{ border-color:var(--accent); color:var(--accent); }}
.count-line {{ color:var(--muted); font-size:13px; margin:12px 2px; }}
.emo-section .bar-label {{ font-size:14px; }}
.agree {{ color:var(--fg); margin:8px 0 16px; }}
.foot {{ color:var(--muted); font-size:12px; margin-top:40px; text-align:center; }}
a {{ color:var(--accent); }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <div class="overline">MBAX 6418 · Assignment 1</div>
    <h1>{esc(data["title"])}</h1>
    <p>{esc(data["subtitle"])}</p>
  </header>

  <div class="cards">
    <div class="kpi"><div class="v">{m["n"]:,}</div><div class="l">Reviews</div></div>
    <div class="kpi"><div class="v">{m["n_pred"]:,}</div><div class="l">Classified</div></div>
    <div class="kpi accent"><div class="v">{m["acc_pct"]}%</div><div class="l">Accuracy</div></div>
    <div class="kpi"><div class="v">{m["n"] - m["n_pred"]}</div><div class="l">API / parse failures</div></div>
  </div>

  {runs_html}

  <section class="card">
    <h2 class="sec-title">📊 Star-rating distribution</h2>
    <p class="sub">Left: the <i>whole file</i> ({whole.get("total", 0):,} Gift Card reviews), heavily 5★-skewed. Right: this balanced run's sample (50 per class), which is not skewed.</p>
    <div class="grid2">
      {whole_panel}
      <div class="panel">{star_bars}<div class="panel-title">Balanced sample ({m["n"]} reviews)</div></div>
    </div>
  </section>

  <section class="card">
    <h2 class="sec-title">🎯 Correct answer vs. model prediction</h2>
    <p class="sub">Left: labels derived from the star rating. Right: what the model (seeing only title+text) predicted.</p>
    <div class="grid2">
      <div class="panel">{answer_chart}<div class="panel-title">Correct answer (from rating)</div></div>
      <div class="panel">{pred_chart}<div class="panel-title">Model prediction</div></div>
    </div>
  </section>

  <section class="card">
    <h2 class="sec-title">✅ Accuracy by class</h2>
    <p class="sub">Percent of reviews in each true class the model labeled correctly.</p>
    {acc_bars}
  </section>

  <section class="card">
    <h2 class="sec-title">🔄 Confusion matrix</h2>
    <p class="sub">Rows = true class (from rating), columns = predicted. Cells on the diagonal are correct. A gray <b>0</b> is a mistake that never happened — e.g. no negative review was ever called positive.</p>
    <table class="conf"><thead>{conf_head}</thead>
    <tbody>{conf_rows}</tbody></table>
  </section>

  {emo_panel}

  <section class="card">
    <h2 class="sec-title">🔎 Explore the reviews</h2>
    <p class="sub">Filter the table; the count updates live.</p>
    <div class="filters">
      <div class="chips" id="filterChips">
        <span class="chip on" data-f="all">All</span>
        <span class="chip" data-f="correct">Correct only</span>
        <span class="chip" data-f="wrong">Wrong only</span>
        <span class="chip" data-f="unscored">Unscored</span>
      </div>
      <label>True&nbsp;<select id="fAnswer">
        <option value="">any</option><option>POSITIVE</option><option>NEUTRAL</option><option>NEGATIVE</option>
      </select></label>
      <label>Predicted&nbsp;<select id="fPred">
        <option value="">any</option><option>POSITIVE</option><option>NEUTRAL</option><option>NEGATIVE</option>
      </select></label>
      <label><input id="fSearch" type="search" placeholder="search title/text…"></label>
    </div>
    <div class="count-line">Showing <b id="shownCount">0</b> / <span id="totalCount">0</span> reviews</div>
    <div class="table-scroll" style="overflow-x:auto;max-height:560px;overflow-y:auto;">
      <table id="reviewTable">
        <thead><tr><th>★</th><th>Review</th><th>True</th><th>Pred</th><th>LLM🧠</th><th>Word📖</th><th>Status</th></tr></thead>
        <tbody id="tbody">{table_body}</tbody>
      </table>
    </div>
  </section>

  <div class="foot">MBAX 6418 · Assignment 1 · sentiment &amp; emotion classifier · single-file offline dashboard</div>
</div>

<script>
const DATA = {js_data};
let curFilter = 'all';

function applyFilters() {{
  const fAnswer = document.getElementById('fAnswer').value;
  const fPred = document.getElementById('fPred').value;
  const q = (document.getElementById('fSearch').value||'').toLowerCase();
  let n = 0;
  for (const r of DATA) {{
    let show = true;
    if (curFilter==='correct') show = r.correct===true;
    else if (curFilter==='wrong') show = r.correct===false;
    else if (curFilter==='unscored') show = r.correct===null || r.correct===undefined;
    if (show && fAnswer && classOf(r.rating)!==fAnswer) show=false;
    if (show && fPred && r.predicted!==fPred) show=false;
    if (show && q) {{
      const hay = ((r.title||'')+' '+(r.text||'')).toLowerCase();
      if (!hay.includes(q)) show=false;
    }}
    const tr = document.getElementById('row-'+r._i);
    if (tr) {{ tr.style.display = show?'' : 'none'; if (show) n++; }}
  }}
  document.getElementById('shownCount').textContent = n;
}}
function classOf(r){{ return r>=4?'POSITIVE':(r===3?'NEUTRAL':'NEGATIVE'); }}
document.querySelectorAll('#filterChips .chip').forEach(ch=>{{
  ch.addEventListener('click',()=>{{
    document.querySelectorAll('#filterChips .chip').forEach(c=>c.classList.remove('on'));
    ch.classList.add('on'); curFilter=ch.dataset.f; applyFilters();
  }});
}});
['fAnswer','fPred'].forEach(id=>document.getElementById(id).addEventListener('change',applyFilters));
document.getElementById('fSearch').addEventListener('input',applyFilters);
document.getElementById('totalCount').textContent = DATA.length;
applyFilters();
</script>
</body>
</html>"""

    return html_doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="predictions jsonl (with nrc if available)")
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--title", default="Amazon Gift Card Review · Sentiment & Emotion")
    ap.add_argument("--subtitle", default="LLM sentiment + emotion prediction checked against the star-rating answer.")
    args = ap.parse_args()

    rows = read_predictions(args.predictions)
    # attach an index for row referencing
    for i, r in enumerate(rows):
        r["_i"] = i
    m = build_metrics(rows)
    doc = build_html({"rows": rows, "metrics": m, "title": args.title, "subtitle": args.subtitle})
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"Accuracy {m['acc_pct']}% · wrote {args.out} ({os.path.getsize(args.out):,} bytes)")


if __name__ == "__main__":
    main()
