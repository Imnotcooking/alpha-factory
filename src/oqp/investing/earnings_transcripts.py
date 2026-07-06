"""RapidAPI earnings-call transcript helpers for discretionary research."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from typing import Any

import pandas as pd

from oqp.investing.news_nlp import score_text_sentiment, sentiment_label, tokenize


RAPIDAPI_EARNINGS_HOST = "earnings-call-transcripts1.p.rapidapi.com"
RapidTranscriptFetcher = Callable[[str, dict[str, Any] | None], Any]

MANAGEMENT_BULLISH_TERMS = {
    "accelerate",
    "backlog",
    "confidence",
    "confident",
    "demand",
    "durable",
    "expand",
    "expansion",
    "growth",
    "improve",
    "improved",
    "momentum",
    "opportunity",
    "raise",
    "raised",
    "record",
    "resilient",
    "strong",
    "upside",
}
MANAGEMENT_BEARISH_TERMS = {
    "cautious",
    "challenge",
    "challenging",
    "decline",
    "delay",
    "headwind",
    "pressure",
    "risk",
    "slowdown",
    "uncertain",
    "uncertainty",
    "weak",
    "weakness",
}
GUIDANCE_TERMS = {
    "guidance",
    "outlook",
    "forecast",
    "expect",
    "expects",
    "target",
    "visibility",
}
MARGIN_TERMS = {"margin", "gross margin", "operating margin", "profitability", "cost", "pricing"}
RISK_TERMS = {"risk", "headwind", "uncertain", "pressure", "slowdown", "competition", "macro"}


def rapidapi_get_json(
    api_key: str | None,
    path: str,
    params: dict[str, Any] | None = None,
    *,
    host: str = RAPIDAPI_EARNINGS_HOST,
    timeout: float = 20.0,
) -> Any:
    """Fetch JSON from the RapidAPI earnings-call-transcripts API."""

    if not api_key:
        return {"error": "Missing RAPID_API_KEY."}
    clean_path = path if path.startswith("/") else f"/{path}"
    url = f"https://{host}{clean_path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": host,
            "Accept": "application/json",
            "User-Agent": "OQP-Earnings-Transcript/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        return {"error": f"HTTP {exc.code}: {body}"}
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}


def payload_data(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        return payload.get("data")
    return payload


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def fetch_rapidapi_earnings_transcript_bundle(
    api_key: str | None,
    symbol: str,
    *,
    fetcher: RapidTranscriptFetcher | None = None,
) -> dict[str, Any]:
    """Fetch latest call metadata, transcript text, summary, and executive segments."""

    symbol_key = symbol.upper().strip()
    if not symbol_key:
        return {"status": "error", "message": "Missing symbol.", "symbol": ""}
    if fetcher is None:
        fetcher = lambda path, params=None: rapidapi_get_json(api_key, path, params)

    latest_payload = fetcher(f"/api/v1/companies/ticker/{symbol_key}/latest", None)
    latest = payload_data(latest_payload)
    if isinstance(latest_payload, dict) and latest_payload.get("error"):
        return {"status": "error", "message": str(latest_payload["error"]), "symbol": symbol_key}
    if not isinstance(latest, dict):
        return {"status": "missing", "message": f"No latest earnings call returned for {symbol_key}.", "symbol": symbol_key}

    earnings_id = latest.get("id") or latest.get("earnings_id") or latest.get("earningsId")
    if not earnings_id:
        return {"status": "missing", "message": f"No earnings id returned for {symbol_key}.", "symbol": symbol_key, "latest": latest}

    summary_payload = fetcher(f"/api/v1/transcripts/{earnings_id}/summary", None)
    full_payload = fetcher(f"/api/v1/transcripts/{earnings_id}", None)
    components_payload = fetcher(f"/api/v1/transcripts/{earnings_id}/components", None)
    speakers_payload = fetcher(f"/api/v1/speakers/{earnings_id}", {"speaker_type": "executive"})

    summary = payload_data(summary_payload)
    full = payload_data(full_payload)
    components = payload_data(components_payload)
    speakers = payload_data(speakers_payload)
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(full, dict):
        full = {}
    if not isinstance(components, dict):
        components = {}
    if not isinstance(speakers, dict):
        speakers = {}

    return {
        "status": "ok",
        "message": "Loaded latest earnings call transcript bundle.",
        "symbol": symbol_key,
        "earnings_id": earnings_id,
        "latest": latest,
        "summary": summary,
        "full": full,
        "components": components,
        "speakers": speakers,
    }


def transcript_text(bundle: dict[str, Any]) -> str:
    full = bundle.get("full") if isinstance(bundle, dict) else {}
    summary = bundle.get("summary") if isinstance(bundle, dict) else {}
    text = ""
    if isinstance(full, dict):
        text = first_text(full, ("full_transcript_text", "transcript", "content", "text"))
    if not text and isinstance(summary, dict):
        text = first_text(summary, ("preview", "summary", "text"))
    return " ".join(text.split())


def executive_segment_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    components = bundle.get("components") if isinstance(bundle, dict) else {}
    if isinstance(components, dict):
        component_rows = components.get("components") or []
        rows: list[dict[str, Any]] = []
        for raw in component_rows if isinstance(component_rows, list) else []:
            if not isinstance(raw, dict):
                continue
            speaker_type = first_text(raw, ("speaker_type", "type", "role")).lower()
            if speaker_type and speaker_type != "executive":
                continue
            text = first_text(raw, ("text", "content", "segment_text", "speech", "transcript_text"))
            if not text:
                continue
            rows.append(
                {
                    "speaker": first_text(raw, ("speaker_name", "speaker", "name")),
                    "title": first_text(raw, ("speaker_title", "title", "role")),
                    "text": " ".join(text.split()),
                }
            )
        if rows:
            return rows

    speakers = bundle.get("speakers") if isinstance(bundle, dict) else {}
    if not isinstance(speakers, dict):
        return []
    segments = speakers.get("segments") or speakers.get("data") or []
    if isinstance(segments, dict):
        segments = segments.get("segments") or [segments]
    rows: list[dict[str, Any]] = []
    for raw in segments if isinstance(segments, list) else []:
        if not isinstance(raw, dict):
            continue
        text = first_text(raw, ("text", "content", "segment_text", "speech", "transcript_text"))
        if not text:
            continue
        speaker = first_text(raw, ("speaker_name", "speaker", "name"))
        title = first_text(raw, ("speaker_title", "title", "role"))
        rows.append({"speaker": speaker, "title": title, "text": " ".join(text.split())})
    return rows


def keyword_hit_text(text: str, terms: set[str], *, limit: int = 8) -> str:
    lowered = text.lower()
    hits = [term for term in sorted(terms) if term in lowered]
    return ", ".join(hits[:limit])


def extract_evidence_snippet(text: str, terms: set[str], *, max_chars: int = 220) -> str:
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        lowered = sentence.lower()
        if any(term in lowered for term in terms):
            snippet = " ".join(sentence.split())
            return snippet[:max_chars]
    return " ".join(text.split())[:max_chars]


def management_score(text: str) -> tuple[float, list[str], list[str]]:
    base_score, bullish, bearish = score_text_sentiment(text)
    tokens = set(tokenize(text))
    bullish_extra = sorted(tokens & MANAGEMENT_BULLISH_TERMS)
    bearish_extra = sorted(tokens & MANAGEMENT_BEARISH_TERMS)
    extra_score = (len(bullish_extra) - len(bearish_extra)) / max(len(bullish_extra) + len(bearish_extra), 6)
    score = max(min((base_score * 0.65) + (extra_score * 0.35), 1.0), -1.0)
    return score, list(dict.fromkeys([*bullish, *bullish_extra])), list(dict.fromkeys([*bearish, *bearish_extra]))


def build_management_tone_frame(bundle: dict[str, Any]) -> pd.DataFrame:
    """Build a management-tone evidence table from transcript text and speaker segments."""

    if not isinstance(bundle, dict) or bundle.get("status") != "ok":
        return pd.DataFrame(columns=["Lens", "Tone", "Score", "Evidence", "Bullish Terms", "Bearish Terms"])

    full_text = transcript_text(bundle)
    executive_rows = executive_segment_rows(bundle)
    executive_text = " ".join(row["text"] for row in executive_rows)
    source_text = executive_text or full_text
    rows: list[dict[str, Any]] = []

    lens_inputs = [
        ("Management Overall", source_text, MANAGEMENT_BULLISH_TERMS | MANAGEMENT_BEARISH_TERMS),
        ("Guidance / Outlook", source_text, GUIDANCE_TERMS),
        ("Margins / Profitability", source_text, MARGIN_TERMS),
        ("Risks / Headwinds", source_text, RISK_TERMS),
    ]
    for lens, text, evidence_terms in lens_inputs:
        score, bullish, bearish = management_score(text)
        rows.append(
            {
                "Lens": lens,
                "Tone": sentiment_label(score),
                "Score": round(float(score), 3),
                "Evidence": extract_evidence_snippet(text, evidence_terms),
                "Bullish Terms": ", ".join(bullish[:8]),
                "Bearish Terms": ", ".join(bearish[:8]),
            }
        )

    return pd.DataFrame(rows)


def build_management_keyword_frame(bundle: dict[str, Any], *, limit: int = 12) -> pd.DataFrame:
    text = " ".join(row["text"] for row in executive_segment_rows(bundle)) or transcript_text(bundle)
    tokens = [token for token in tokenize(text) if len(token) > 3]
    rows = [{"Keyword": token, "Count": count} for token, count in Counter(tokens).most_common(limit)]
    return pd.DataFrame(rows, columns=["Keyword", "Count"])


def management_tone_summary(bundle: dict[str, Any], tone_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    if not isinstance(bundle, dict) or bundle.get("status") != "ok":
        return {"status": bundle.get("status", "error") if isinstance(bundle, dict) else "error"}
    if tone_frame is None:
        tone_frame = build_management_tone_frame(bundle)
    score = 0.0
    if isinstance(tone_frame, pd.DataFrame) and not tone_frame.empty:
        score = float(pd.to_numeric(tone_frame["Score"], errors="coerce").mean())
    latest = bundle.get("latest") if isinstance(bundle.get("latest"), dict) else {}
    executive_segments = len(executive_segment_rows(bundle))
    return {
        "status": "ok",
        "tone": sentiment_label(score),
        "score": score,
        "event_date": latest.get("event_date_time") or "",
        "title": latest.get("transcript_title") or "",
        "earnings_id": bundle.get("earnings_id"),
        "executive_segments": executive_segments,
        "tone_source": "executive components" if executive_segments else "full transcript",
    }
