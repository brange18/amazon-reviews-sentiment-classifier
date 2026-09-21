"""
MBAX 6418 - Assignment 1
Core classifier module.

Exposes a reusable prompt + LLM call that classifies an Amazon Gift Card review
as POSITIVE or NEGATIVE based ONLY on its title and text (never the rating).

The output is a strict JSON object so callers can read it programmatically:
    {"sentiment": "POSITIVE" | "NEGATIVE"}
"""

from openai import OpenAI

# --- Fixed endpoint / model -------------------------------------------------
# Base URL and key provided by the class; model served on the class endpoint.
LLM_BASE_URL = "http://dobolyi.com:9001/v1"
LLM_API_KEY = "6418"
LLM_MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

# Token budget: the served model is a "thinking" model and may reason before
# answering. Give it comfortable headroom so it always finishes its answer.
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.0

# Persistent cache of raw model replies, keyed by a hash of
# (title, text, n_classes, with_emotion). Lets interrupted runs resume without
# re-issuing the same model call twice.
import hashlib
import json as _json
import os as _os

_CACHE_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            "results", "llm_cache.jsonl")


def _cache_key(title, text, n_classes, with_emotion):
    payload = f"{title}\x00{text}\x00{n_classes}\x00{with_emotion}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache():
    cache = {}
    if _os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                    cache[rec["key"]] = rec["raw"]
                except Exception:  # noqa: BLE001
                    continue
    return cache


def _append_cache(key, raw):
    _os.makedirs(_os.path.dirname(_CACHE_FILE), exist_ok=True)
    with open(_CACHE_FILE, "a", encoding="utf-8") as f:
        f.write(_json.dumps({"key": key, "raw": raw}) + "\n")


# --- The prompt -------------------------------------------------------------
SYSTEM_PROMPT_2CLASS = (
    "You are a review-sentiment classifier. You classify Amazon product "
    "reviews. You will be given a review's TITLE and BODY. Classify the "
    "overall sentiment a reader takes away from the review as either "
    "POSITIVE or NEGATIVE.\n\n"
    "Rules:\n"
    "- Return only one class: POSITIVE or NEGATIVE.\n"
    "- Base your judgment solely on the TITLE and BODY you are given.\n"
    "- Do NOT invent facts, star ratings, or scores that are not in the text.\n"
    "- If the review expresses satisfaction, praise, likeliness to buy again, "
    "or a generally favorable feel, that is POSITIVE.\n"
    "- If the review expresses disappointment, a problem, a complaint, "
    "frustration, or a generally unfavorable feel, that is NEGATIVE.\n"
    "- Decide edge cases yourself (e.g. a positive title with a negative body, "
    "or a terse/angry short review) by weighing the strongest overall signal.\n"
    '- Reply with ONLY a JSON object of the form {"sentiment": "POSITIVE"} '
    'or {"sentiment": "NEGATIVE"}. No extra text, no explanation.'
)

SYSTEM_PROMPT_3CLASS = (
    "You are a review-sentiment classifier. You classify Amazon product "
    "reviews. You will be given a review's TITLE and BODY. Classify the "
    "overall sentiment a reader takes away from the review as one of "
    "POSITIVE, NEUTRAL, or NEGATIVE.\n\n"
    "Rules:\n"
    "- Return only one class: POSITIVE, NEUTRAL, or NEGATIVE.\n"
    "- Base your judgment solely on the TITLE and BODY you are given.\n"
    "- Do NOT invent facts, star ratings, or scores that are not in the text.\n"
    "- POSITIVE: satisfaction, praise, likeliness to buy again, generally "
    "favorable feel.\n"
    "- NEGATIVE: disappointment, a problem, a complaint, frustration, "
    "generally unfavorable feel.\n"
    "- NEUTRAL: balanced, factual, mixed-positive-and-negative, or neither "
    "clearly favorable nor unfavorable.\n"
    "- A three-star-vibe but lukewarm review with both good and bad points "
    "should usually be NEUTRAL. Only call it POSITIVE or NEGATIVE if the "
    "weight of evidence clearly favors one side.\n"
    "- Decide edge cases yourself (e.g. a positive title with a negative body, "
    "or a terse/angry short review) by weighing the strongest overall signal.\n"
    '- Reply with ONLY a JSON object of the form {"sentiment": "NEUTRAL"}. '
    "No extra text, no explanation."
)

# Emotion list appended to the prompt when primary emotion is requested.
EMOTION_SPEC = (
    "\n\nAdditionally, identify the PRIMARY EMOTION the review most strongly "
    "expresses, chosen from exactly these eight: anger, anticipation, disgust, "
    "fear, joy, sadness, surprise, trust.\n"
    "- Pick the single dominant emotion. If none stands out, choose the best "
    "fit even if imperfect.\n"
    '- Final reply MUST be ONLY a JSON object like '
    '{"sentiment": "POSITIVE", "emotion": "joy"}. No extra text.'
)

EMOTION_LABELS = ["anger", "anticipation", "disgust", "fear", "joy",
                  "sadness", "surprise", "trust"]


def get_system_prompt(n_classes=2, with_emotion=False):
    base = SYSTEM_PROMPT_3CLASS if n_classes == 3 else SYSTEM_PROMPT_2CLASS
    if with_emotion:
        base = base + EMOTION_SPEC
    return base


def build_user_prompt(title: str, text: str) -> str:
    """Build the user message for one review from its title and text."""
    title = (title or "").strip()
    text = (text or "").strip()
    return (
        "Review to classify:\n"
        f"TITLE: {title}\n"
        f"BODY: {text}\n\n"
        "Classify the sentiment."
    )


# --- LLM plumbing ------------------------------------------------------------
def get_client() -> OpenAI:
    """Return a configured OpenAI-compatible client for the class endpoint."""
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def call_llm(client: OpenAI, title: str, text: str, max_tokens: int = None,
             n_retries: int = 3, n_classes=2, with_emotion=False):
    """Call the model once and return the raw text of its reply.

    Replays a cached reply when the same (title, text, settings) was already
    answered, so interrupted runs don't re-issue model calls.
    """
    import time

    key = _cache_key(title, text, n_classes, with_emotion)
    cache = _load_cache()
    if key in cache:
        return cache[key]

    kwargs = dict(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": get_system_prompt(n_classes, with_emotion)},
            {"role": "user", "content": build_user_prompt(title, text)},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=max_tokens or LLM_MAX_TOKENS,
    )
    for attempt in range(1, n_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            if content and content.strip():
                _append_cache(key, content)
                return content
        except Exception as exc:  # noqa: BLE001
            if attempt == n_retries:
                raise
            time.sleep(2 * attempt)
    raise RuntimeError("Model returned an empty reply after all retries.")


# --- Parsing ----------------------------------------------------------------
def _strict_json_obj(text: str):
    """Return a dict if `text` is a single JSON object, else None."""
    import json

    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_sentiment(obj):
    """Return a normalized POSITIVE/NEGATIVE/NEUTRAL from a parsed dict, or None."""
    if not isinstance(obj, dict):
        return None
    for key in ("sentiment", "Sentiment", "label"):
        val = obj.get(key)
        if isinstance(val, str):
            upper = val.strip().upper()
            if upper in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
                return upper
    return None


def _extract_emotion(obj):
    """Return an allowed emotion label from a parsed dict, or None."""
    if not isinstance(obj, dict):
        return None
    for key in ("emotion", "Emotion", "primary_emotion"):
        val = obj.get(key)
        if isinstance(val, str):
            emo = val.strip().lower()
            if emo in EMOTION_LABELS:
                return emo
    return None


def parse_sentiment(raw: str, with_emotion=False):
    """Parse the model reply into sentiment (and optionally emotion).

    Tolerates a bit of noise: extra prose, code fences, or single-quoted JSON.
    Raises ValueError if nothing parseable survives.
    """
    import re

    if not raw:
        raise ValueError("empty model reply")

    def try_obj(obj):
        """Extract from a dict, returning (sentiment or None, emotion or None)."""
        sent = _extract_sentiment(obj)
        emo = _extract_emotion(obj) if with_emotion else None
        return sent, emo

    # 1) Strict JSON path.
    obj = _strict_json_obj(raw)
    if obj is not None:
        sent, emo = try_obj(obj)
        if sent is not None:
            out = {"sentiment": sent}
            if with_emotion:
                out["emotion"] = emo
            return out

    # 2) Fall back: strip code fences and pull the first { ... } block.
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        obj = _strict_json_obj(m.group(0))
        if obj is not None:
            sent, emo = try_obj(obj)
            if sent is not None:
                out = {"sentiment": sent}
                if with_emotion:
                    out["emotion"] = emo
                return out

    # 3) Last resort: scan for a bare class label.
    if re.search(r"\bPOSITIVE\b", raw):
        return {"sentiment": "POSITIVE"}
    if re.search(r"\bNEGATIVE\b", raw):
        return {"sentiment": "NEGATIVE"}
    if re.search(r"\bNEUTRAL\b", raw):
        return {"sentiment": "NEUTRAL"}

    raise ValueError(f"could not parse sentiment from model reply: {raw!r}"[:500])


# --- Convenience ------------------------------------------------------------
def classify_review(client: OpenAI, title: str, text: str, n_classes=2, with_emotion=False):
    """Classify a single review, returning the parsed result dict."""
    raw = call_llm(client, title, text, n_classes=n_classes, with_emotion=with_emotion)
    return parse_sentiment(raw, with_emotion=with_emotion)


if __name__ == "__main__":
    import json

    c = get_client()
    sample_reviews = [
        # obviously positive
        ("Great gift", "Having Amazon money is always good."),
        ("Love it!", "Fast delivery, easy to load, perfect for a gift."),
        # obviously negative
        ("Waste of money", "Card never worked and support was useless."),
        ("Terrible", "Tried to use it and it was declined everywhere."),
        # edge cases: positive title / negative body
        ("Great looking card", "But it would not load, very frustrating."),
        # terse/conflicting
        ("Ugh", "so disappointed."),
    ]
    print(json.dumps({"endpoint": LLM_MODEL}, indent=2))
    for title, text in sample_reviews:
        try:
            res = classify_review(c, title, text)
        except Exception as exc:  # noqa: BLE001
            res = {"error": str(exc)[:200]}
        print("----")
        print(f"TITLE : {title}")
        print(f"BODY  : {text}")
        print(f"CLASS : {res}")
