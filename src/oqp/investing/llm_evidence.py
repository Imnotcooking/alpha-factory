"""LLM-backed evidence synthesis for discretionary investing workflows."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import urllib.error
import urllib.request
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from oqp.config.paths import REPO_ROOT
from oqp.investing.earnings_transcripts import executive_segment_rows, transcript_text


DEFAULT_LLM_EVIDENCE_DB_PATH = REPO_ROOT / "runtime" / "db" / "investing" / "llm_evidence.db"

LLMChatClient = Callable[[list[dict[str, str]], dict[str, Any]], str | dict[str, Any]]


def ensure_llm_evidence_schema(path: str | Path = DEFAULT_LLM_EVIDENCE_DB_PATH) -> Path:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_outlook_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                analysis_json TEXT NOT NULL,
                raw_response TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_llm_outlook_cache_lookup
            ON llm_outlook_cache (symbol, provider, model, source_hash, created_at)
            """
        )
        conn.commit()
    return db_path


def compact_text(value: Any, *, max_chars: int | None = None) -> str:
    text = " ".join(str(value or "").split())
    if max_chars is not None and len(text) > max_chars:
        return f"{text[:max_chars].rstrip()}..."
    return text


def safe_json_loads(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM response did not contain a JSON object.")
    return parsed


def normalize_chat_base_url(base_url: str) -> str:
    clean = (base_url or "").strip().rstrip("/")
    if not clean:
        clean = "https://api.z.ai/api/paas/v4"
    if clean.endswith("/chat/completions"):
        return clean
    return f"{clean}/chat/completions"


def openai_compatible_chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: int = 60,
    temperature: float = 0.1,
    max_tokens: int = 2200,
) -> str:
    url = normalize_chat_base_url(base_url)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "OQP-LLM-Evidence/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"LLM API HTTP {exc.code}: {body}") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"LLM API request failed: {exc}") from exc

    choices = result.get("choices") if isinstance(result, dict) else None
    if not choices:
        raise RuntimeError("LLM API returned no choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text") or part.get("content") or "")
            for part in content
            if isinstance(part, dict)
        )
    if not content:
        raise RuntimeError("LLM API response had no message content.")
    return str(content)


def _article_rows(articles: pd.DataFrame | None, *, limit: int = 14) -> list[dict[str, str]]:
    if articles is None or articles.empty:
        return []
    frame = articles.copy().head(limit)
    rows: list[dict[str, str]] = []
    for row in frame.to_dict("records"):
        rows.append(
            {
                "published_at": compact_text(row.get("Published") or row.get("published_at")),
                "source": compact_text(row.get("Site") or row.get("site") or row.get("source")),
                "title": compact_text(row.get("Title") or row.get("title"), max_chars=180),
                "text": compact_text(row.get("Text") or row.get("text"), max_chars=320),
            }
        )
    return rows


def _valuation_snapshot(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    keys = (
        "symbol",
        "companyName",
        "price",
        "marketCap",
        "pe",
        "pegRatio",
        "sector",
        "industry",
        "freeCashFlowTTM",
        "revenuePerShareTTM",
        "returnOnCapitalEmployedTTM",
        "debtToEquity",
    )
    return {key: data.get(key) for key in keys if data.get(key) not in (None, "")}


def build_company_outlook_packet(
    *,
    symbol: str,
    transcript_bundle: dict[str, Any] | None,
    news_articles: pd.DataFrame | None,
    valuation_data: dict[str, Any] | None,
) -> dict[str, Any]:
    bundle = transcript_bundle if isinstance(transcript_bundle, dict) else {}
    latest = bundle.get("latest") if isinstance(bundle.get("latest"), dict) else {}
    exec_rows = executive_segment_rows(bundle)
    management_segments = [
        {
            "speaker": compact_text(row.get("speaker")),
            "title": compact_text(row.get("title")),
            "text": compact_text(row.get("text"), max_chars=900),
        }
        for row in exec_rows[:18]
    ]
    fallback_transcript = transcript_text(bundle)
    return {
        "symbol": symbol.upper().strip(),
        "latest_call": {
            "earnings_id": bundle.get("earnings_id"),
            "event_date": latest.get("event_date_time") or latest.get("date"),
            "title": latest.get("transcript_title") or latest.get("title"),
        },
        "management_segments": management_segments,
        "transcript_excerpt": compact_text(fallback_transcript, max_chars=10_000)
        if not management_segments
        else "",
        "recent_news": _article_rows(news_articles),
        "valuation_snapshot": _valuation_snapshot(valuation_data),
    }


def packet_hash(packet: dict[str, Any]) -> str:
    encoded = json.dumps(packet, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_outlook_messages(packet: dict[str, Any]) -> list[dict[str, str]]:
    schema = {
        "executive_summary": "2-4 sentence synthesis of what matters.",
        "short_term_outlook": {
            "horizon": "0-3 months",
            "label": "bullish | neutral | bearish | mixed | insufficient_data",
            "confidence": 0.0,
            "summary": "near-term catalysts, earnings reaction, options-relevant setup",
            "evidence": ["short evidence item with source/speaker if available"],
        },
        "mid_term_outlook": {
            "horizon": "3-18 months",
            "label": "bullish | neutral | bearish | mixed | insufficient_data",
            "confidence": 0.0,
            "summary": "growth, margins, demand and execution trajectory",
            "evidence": ["short evidence item with source/speaker if available"],
        },
        "long_term_outlook": {
            "horizon": "18+ months",
            "label": "bullish | neutral | bearish | mixed | insufficient_data",
            "confidence": 0.0,
            "summary": "moat, terminal growth, durability, structural risks",
            "evidence": ["short evidence item with source/speaker if available"],
        },
        "dcf_assumption_clues": {
            "wacc": "risk-quality comment, not a numeric overwrite",
            "y1_5_growth": "explicit growth clue",
            "y6_10_growth": "fade / durability clue",
            "terminal_growth": "long-run durability clue",
            "margin": "margin or FCF conversion clue",
        },
        "options_implications": {
            "directional_bias": "bullish | neutral | bearish | mixed | insufficient_data",
            "volatility_bias": "long_vol | short_vol | neutral | insufficient_data",
            "watch_catalysts": ["catalyst or date to watch"],
        },
        "key_risks": ["risk"],
        "source_limits": ["what the source packet cannot prove"],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an equity research evidence assistant. Use only the supplied packet. "
                "Do not invent facts. Return strict JSON only, with no markdown or commentary."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": (
                        "Synthesize short-term, mid-term, and long-term outlook evidence for a stock "
                        "research dashboard. Keep conclusions cautious and evidence-linked."
                    ),
                    "required_schema": schema,
                    "source_packet": packet,
                },
                ensure_ascii=False,
            ),
        },
    ]


def load_cached_llm_outlook(
    *,
    symbol: str,
    provider: str,
    model: str,
    source_hash: str,
    path: str | Path = DEFAULT_LLM_EVIDENCE_DB_PATH,
    max_age_hours: float | None = 168.0,
) -> dict[str, Any] | None:
    db_path = ensure_llm_evidence_schema(path)
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            """
            SELECT analysis_json, raw_response, created_at
            FROM llm_outlook_cache
            WHERE symbol = ? AND provider = ? AND model = ? AND source_hash = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (symbol.upper().strip(), provider, model, source_hash),
        ).fetchone()
    if row is None:
        return None
    created_at = datetime.fromisoformat(str(row[2]))
    if max_age_hours is not None:
        if datetime.now(timezone.utc) - created_at > timedelta(hours=float(max_age_hours)):
            return None
    return {
        "analysis": json.loads(row[0]),
        "raw_response": row[1],
        "created_at": row[2],
    }


def write_llm_outlook(
    *,
    symbol: str,
    provider: str,
    model: str,
    source_hash: str,
    analysis: dict[str, Any],
    raw_response: str,
    path: str | Path = DEFAULT_LLM_EVIDENCE_DB_PATH,
    created_at: datetime | None = None,
) -> Path:
    db_path = ensure_llm_evidence_schema(path)
    created_text = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(
        timespec="seconds"
    )
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO llm_outlook_cache (
                symbol, provider, model, source_hash, analysis_json, raw_response, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol.upper().strip(),
                provider,
                model,
                source_hash,
                json.dumps(analysis, sort_keys=True, ensure_ascii=False),
                raw_response,
                created_text,
            ),
        )
        conn.commit()
    return db_path


def analyze_company_outlook(
    *,
    symbol: str,
    api_key: str | None,
    provider: str,
    base_url: str,
    model: str,
    transcript_bundle: dict[str, Any] | None,
    news_articles: pd.DataFrame | None,
    valuation_data: dict[str, Any] | None,
    path: str | Path = DEFAULT_LLM_EVIDENCE_DB_PATH,
    max_age_hours: float | None = 168.0,
    timeout_seconds: int = 60,
    force_refresh: bool = False,
    chat_client: LLMChatClient | None = None,
) -> dict[str, Any]:
    symbol_key = symbol.upper().strip()
    if not symbol_key:
        return {"status": "error", "message": "Missing symbol."}
    if not api_key:
        return {"status": "error", "message": "Missing LLM API key."}

    packet = build_company_outlook_packet(
        symbol=symbol_key,
        transcript_bundle=transcript_bundle,
        news_articles=news_articles,
        valuation_data=valuation_data,
    )
    source_hash = packet_hash(packet)
    if not force_refresh:
        cached = load_cached_llm_outlook(
            symbol=symbol_key,
            provider=provider,
            model=model,
            source_hash=source_hash,
            path=path,
            max_age_hours=max_age_hours,
        )
        if cached is not None:
            return {
                "status": "cached",
                "message": "Loaded cached LLM outlook.",
                "symbol": symbol_key,
                "provider": provider,
                "model": model,
                "source_hash": source_hash,
                **cached,
            }

    messages = build_outlook_messages(packet)
    try:
        if chat_client is None:
            raw_response = openai_compatible_chat_completion(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                timeout_seconds=timeout_seconds,
            )
        else:
            raw = chat_client(messages, packet)
            raw_response = json.dumps(raw, ensure_ascii=False) if isinstance(raw, dict) else str(raw)
        analysis = safe_json_loads(raw_response)
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "symbol": symbol_key,
            "provider": provider,
            "model": model,
            "source_hash": source_hash,
        }

    db_path = write_llm_outlook(
        symbol=symbol_key,
        provider=provider,
        model=model,
        source_hash=source_hash,
        analysis=analysis,
        raw_response=raw_response,
        path=path,
    )
    return {
        "status": "ok",
        "message": "Generated and cached LLM outlook.",
        "symbol": symbol_key,
        "provider": provider,
        "model": model,
        "source_hash": source_hash,
        "analysis": analysis,
        "raw_response": raw_response,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "db_path": str(db_path),
    }
