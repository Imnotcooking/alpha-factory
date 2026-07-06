"""News/article cache and lightweight NLP for discretionary stock research."""

from __future__ import annotations

import math
import re
import sqlite3
from importlib.util import find_spec
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from oqp.config.paths import REPO_ROOT
from oqp.investing.stock_valuation import fetch_fmp_json


DEFAULT_NEWS_NLP_DB_PATH = REPO_ROOT / "runtime" / "db" / "investing" / "news_nlp.db"
DEFAULT_NEWS_CACHE_MAX_AGE_HOURS = 12.0

FMPNewsFetcher = Callable[..., Any]

BULLISH_TERMS = {
    "accelerate",
    "approval",
    "beat",
    "beats",
    "breakthrough",
    "bullish",
    "buyback",
    "contract",
    "demand",
    "expand",
    "growth",
    "improve",
    "improved",
    "launch",
    "margin",
    "outperform",
    "profit",
    "raise",
    "raised",
    "record",
    "recover",
    "recovery",
    "strong",
    "upgrade",
    "upside",
}
BULLISH_PHRASES = {
    "beats expectations",
    "raises guidance",
    "price target raised",
    "strong demand",
    "record revenue",
    "share repurchase",
}
BEARISH_TERMS = {
    "bearish",
    "cut",
    "decline",
    "downgrade",
    "fall",
    "falls",
    "fraud",
    "investigation",
    "lawsuit",
    "loss",
    "miss",
    "misses",
    "pressure",
    "probe",
    "recall",
    "recession",
    "risk",
    "slowdown",
    "slump",
    "weak",
    "weakness",
    "warn",
    "warning",
}
BEARISH_PHRASES = {
    "cuts guidance",
    "misses expectations",
    "price target cut",
    "regulatory probe",
    "supply chain disruption",
    "weaker demand",
}
STOP_WORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "amid",
    "among",
    "and",
    "are",
    "around",
    "because",
    "been",
    "before",
    "being",
    "but",
    "can",
    "company",
    "could",
    "from",
    "has",
    "have",
    "into",
    "its",
    "more",
    "new",
    "not",
    "over",
    "said",
    "says",
    "share",
    "shares",
    "stock",
    "than",
    "that",
    "the",
    "their",
    "this",
    "through",
    "under",
    "with",
    "would",
    "year",
    "years",
}
TOPIC_KEYWORDS = {
    "Earnings": {"earnings", "eps", "revenue", "profit", "margin", "guidance", "quarter", "forecast"},
    "Analyst Action": {"analyst", "upgrade", "downgrade", "rating", "target", "outperform", "underperform"},
    "Product / Demand": {"launch", "product", "demand", "orders", "customer", "sales", "shipment"},
    "Regulatory / Legal": {"regulatory", "lawsuit", "probe", "investigation", "approval", "ban", "fine"},
    "Macro / Rates": {"rates", "fed", "inflation", "recession", "tariff", "china", "dollar", "oil"},
    "Deal / Capital": {"merger", "acquisition", "buyback", "debt", "offering", "dividend", "stake"},
}


@dataclass(frozen=True, slots=True)
class NewsNLPResult:
    """Dashboard-ready article NLP output."""

    symbol: str
    article_frame: pd.DataFrame
    keyword_frame: pd.DataFrame
    topic_frame: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SentimentProviderStatus:
    """Readiness row for a news/sentiment provider."""

    provider: str
    role: str
    status: str
    detail: str

    def as_row(self) -> dict[str, str]:
        return {
            "Provider": self.provider,
            "Role": self.role,
            "Status": self.status,
            "Detail": self.detail,
        }


def ensure_news_nlp_schema(path: str | Path = DEFAULT_NEWS_NLP_DB_PATH) -> Path:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_articles (
                symbol TEXT NOT NULL,
                published_at TEXT NOT NULL,
                title TEXT NOT NULL,
                site TEXT,
                url TEXT NOT NULL,
                text TEXT,
                source TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (symbol, url, published_at, title)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_news_articles_symbol_published
            ON news_articles (symbol, published_at)
            """
        )
        conn.commit()
    return db_path


def load_cached_news_articles(
    symbol: str,
    *,
    path: str | Path = DEFAULT_NEWS_NLP_DB_PATH,
    limit: int = 80,
) -> pd.DataFrame:
    symbol_key = normalize_symbol(symbol)
    if not symbol_key:
        return empty_news_frame()
    db_path = ensure_news_nlp_schema(path)
    with closing(sqlite3.connect(db_path)) as conn:
        frame = pd.read_sql_query(
            """
            SELECT symbol, published_at, title, site, url, text, source, fetched_at
            FROM news_articles
            WHERE symbol = ?
            ORDER BY published_at DESC, fetched_at DESC
            LIMIT ?
            """,
            conn,
            params=(symbol_key, int(limit)),
        )
    return frame if not frame.empty else empty_news_frame()


def write_news_articles(
    symbol: str,
    articles: pd.DataFrame,
    *,
    path: str | Path = DEFAULT_NEWS_NLP_DB_PATH,
    source: str = "FMP",
    fetched_at: datetime | None = None,
) -> int:
    symbol_key = normalize_symbol(symbol)
    if not symbol_key or articles is None or articles.empty:
        return 0
    normalized = normalize_news_articles(articles, symbol_key=symbol_key, source=source)
    if normalized.empty:
        return 0
    fetched_text = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(
        timespec="seconds"
    )
    rows = [
        (
            symbol_key,
            row["published_at"],
            row["title"],
            row["site"],
            row["url"],
            row["text"],
            row["source"],
            fetched_text,
        )
        for row in normalized.to_dict("records")
    ]
    db_path = ensure_news_nlp_schema(path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executemany(
            """
            INSERT INTO news_articles (
                symbol, published_at, title, site, url, text, source, fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, url, published_at, title) DO UPDATE SET
                site = excluded.site,
                text = excluded.text,
                source = excluded.source,
                fetched_at = excluded.fetched_at
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def news_cache_status(
    symbol: str,
    *,
    path: str | Path = DEFAULT_NEWS_NLP_DB_PATH,
    max_age_hours: float = DEFAULT_NEWS_CACHE_MAX_AGE_HOURS,
) -> dict[str, Any]:
    symbol_key = normalize_symbol(symbol)
    if not symbol_key:
        return {"symbol": "", "rows": 0, "fetched_at": "", "age_hours": None, "state": "missing"}
    db_path = ensure_news_nlp_schema(path)
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS rows, MAX(fetched_at) AS fetched_at, MAX(published_at) AS latest_published
            FROM news_articles
            WHERE symbol = ?
            """,
            (symbol_key,),
        ).fetchone()
    rows = int(row[0] or 0) if row else 0
    fetched_at = row[1] if row else ""
    age = cache_age_hours(fetched_at)
    state = "missing" if rows == 0 else "stale" if age is None or age > max_age_hours else "fresh"
    return {
        "symbol": symbol_key,
        "rows": rows,
        "fetched_at": fetched_at or "",
        "latest_published": row[2] if row else "",
        "age_hours": age,
        "state": state,
    }


def load_or_refresh_news_articles(
    symbol: str,
    api_key: str | None,
    *,
    path: str | Path = DEFAULT_NEWS_NLP_DB_PATH,
    limit: int = 60,
    max_age_hours: float = DEFAULT_NEWS_CACHE_MAX_AGE_HOURS,
    fmp_fetcher: FMPNewsFetcher = fetch_fmp_json,
) -> pd.DataFrame:
    status = news_cache_status(symbol, path=path, max_age_hours=max_age_hours)
    if status["state"] == "fresh":
        return load_cached_news_articles(symbol, path=path, limit=limit)
    fresh = fetch_fmp_news_articles(api_key, symbol, limit=limit, fmp_fetcher=fmp_fetcher)
    if not fresh.empty:
        write_news_articles(symbol, fresh, path=path, source="FMP")
    return load_cached_news_articles(symbol, path=path, limit=limit)


def fetch_fmp_news_articles(
    api_key: str | None,
    symbol: str,
    *,
    limit: int = 60,
    fmp_fetcher: FMPNewsFetcher = fetch_fmp_json,
) -> pd.DataFrame:
    """Fetch symbol news/articles from known FMP news endpoint shapes."""

    symbol_key = normalize_symbol(symbol)
    if not api_key or not symbol_key:
        return empty_news_frame()
    endpoint_specs = [
        ("stock_news", False, {"tickers": symbol_key, "limit": limit}, "FMP stock_news"),
        ("news/stock", True, {"symbols": symbol_key, "limit": limit}, "FMP stable news/stock"),
        ("news/stock", True, {"symbol": symbol_key, "limit": limit}, "FMP stable news/stock"),
        ("news/articles", True, {"symbols": symbol_key, "limit": limit}, "FMP stable articles"),
        ("stock-news", True, {"symbol": symbol_key, "limit": limit}, "FMP stable stock-news"),
    ]
    frames = []
    for endpoint, stable, params, source in endpoint_specs:
        payload = fmp_fetcher(
            api_key,
            endpoint,
            stable=stable,
            params=params,
            suppress_error_messages=False,
        )
        frame = normalize_news_articles(payload, symbol_key=symbol_key, source=source)
        if not frame.empty:
            frames.append(frame)
            break
    if not frames:
        return empty_news_frame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["url", "published_at", "title"]).head(limit)


def normalize_news_articles(payload: Any, *, symbol_key: str, source: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in iter_article_records(payload):
        title = first_text(raw, ("title", "headline", "newsTitle"))
        url = first_text(raw, ("url", "link", "newsURL"))
        text = first_text(raw, ("text", "content", "summary", "snippet", "description"))
        published_at = normalize_datetime(
            first_text(raw, ("publishedDate", "published_at", "date", "publishedAt", "datetime", "timestamp"))
        )
        if not title and not text:
            continue
        rows.append(
            {
                "symbol": symbol_key,
                "published_at": published_at,
                "title": title or text[:80],
                "site": first_text(raw, ("site", "publisher", "source", "sourceName")),
                "url": url or f"no-url::{published_at}::{title[:60]}",
                "text": text,
                "source": source,
            }
        )
    if not rows:
        return empty_news_frame()
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["url", "published_at", "title"])
    return frame.sort_values("published_at", ascending=False).reset_index(drop=True)


def analyze_news_articles(symbol: str, articles: pd.DataFrame, *, top_n: int = 30) -> NewsNLPResult:
    symbol_key = normalize_symbol(symbol)
    if articles is None or articles.empty:
        return NewsNLPResult(
            symbol=symbol_key,
            article_frame=empty_scored_article_frame(),
            keyword_frame=pd.DataFrame(columns=["Keyword", "Count", "Tone"]),
            topic_frame=pd.DataFrame(columns=["Topic", "Count", "Avg Sentiment"]),
            summary={
                "sentiment_label": "No data",
                "sentiment_score": 0.0,
                "article_count": 0,
                "top_topics": "",
                "top_keywords": "",
            },
        )
    rows = []
    token_counter: Counter[str] = Counter()
    topic_rows: list[dict[str, Any]] = []
    for raw in articles.head(top_n).to_dict("records"):
        combined = f"{raw.get('title') or ''} {raw.get('text') or ''}"
        tokens = tokenize(combined)
        token_counter.update(tokens)
        score, bullish_hits, bearish_hits = score_text_sentiment(combined)
        topics = classify_topics(tokens)
        for topic in topics:
            topic_rows.append({"Topic": topic, "Score": score})
        rows.append(
            {
                "Published": raw.get("published_at") or "",
                "Title": raw.get("title") or "",
                "Site": raw.get("site") or "",
                "Sentiment": sentiment_label(score),
                "Score": score,
                "Topics": ", ".join(topics) if topics else "General",
                "Bullish Terms": ", ".join(bullish_hits[:5]),
                "Bearish Terms": ", ".join(bearish_hits[:5]),
                "URL": raw.get("url") or "",
            }
        )
    article_frame = pd.DataFrame(rows)
    keyword_frame = keyword_summary(token_counter)
    topic_frame = topic_summary(topic_rows)
    average_score = float(pd.to_numeric(article_frame["Score"], errors="coerce").mean()) if not article_frame.empty else 0.0
    summary = {
        "sentiment_label": sentiment_label(average_score),
        "sentiment_score": average_score,
        "article_count": int(len(article_frame)),
        "top_topics": ", ".join(topic_frame["Topic"].head(3).tolist()) if not topic_frame.empty else "",
        "top_keywords": ", ".join(keyword_frame["Keyword"].head(6).tolist()) if not keyword_frame.empty else "",
    }
    return NewsNLPResult(
        symbol=symbol_key,
        article_frame=article_frame,
        keyword_frame=keyword_frame,
        topic_frame=topic_frame,
        summary=summary,
    )


def build_news_catalyst_board(article_frame: pd.DataFrame) -> pd.DataFrame:
    """Group scored news rows into catalyst buckets for dashboard triage."""

    if article_frame is None or article_frame.empty:
        return empty_catalyst_board_frame()
    rows: list[dict[str, Any]] = []
    frame = article_frame.copy()
    frame["Score"] = pd.to_numeric(frame.get("Score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    frame["PublishedParsed"] = pd.to_datetime(frame.get("Published", pd.Series(dtype=str)), errors="coerce", utc=True)
    for topic in sorted({topic for topics in frame.get("Topics", []) for topic in split_topics(topics)}):
        subset = frame[frame["Topics"].map(lambda value, t=topic: t in split_topics(value))].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("PublishedParsed", ascending=False, na_position="last")
        latest = subset.iloc[0]
        avg_score = float(subset["Score"].mean())
        rows.append(
            {
                "Catalyst": topic,
                "Articles": int(len(subset)),
                "Tone": sentiment_label(avg_score),
                "Avg Score": avg_score,
                "Latest": normalize_dashboard_date(latest.get("Published")),
                "Latest Headline": str(latest.get("Title") or ""),
                "Evidence": catalyst_evidence(subset),
                "Action Read": catalyst_action_read(topic, avg_score),
            }
        )
    if not rows:
        return empty_catalyst_board_frame()
    result = pd.DataFrame(rows)
    return result.sort_values(["Articles", "Avg Score"], ascending=[False, False]).reset_index(drop=True)


def build_news_evidence_frame(article_frame: pd.DataFrame, *, limit: int = 8) -> pd.DataFrame:
    """Return the most useful article evidence rows for a thesis draft."""

    if article_frame is None or article_frame.empty:
        return pd.DataFrame(columns=["Published", "Catalyst", "Tone", "Score", "Headline", "Evidence"])
    frame = article_frame.copy()
    frame["Score"] = pd.to_numeric(frame.get("Score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    frame["PublishedParsed"] = pd.to_datetime(frame.get("Published", pd.Series(dtype=str)), errors="coerce", utc=True)
    frame["AbsScore"] = frame["Score"].abs()
    frame = frame.sort_values(["AbsScore", "PublishedParsed"], ascending=[False, False], na_position="last")
    rows = []
    for row in frame.head(limit).to_dict("records"):
        topics = split_topics(row.get("Topics"))
        rows.append(
            {
                "Published": normalize_dashboard_date(row.get("Published")),
                "Catalyst": ", ".join(topics[:2]) if topics else "General",
                "Tone": str(row.get("Sentiment") or sentiment_label(float(row.get("Score") or 0.0))),
                "Score": float(row.get("Score") or 0.0),
                "Headline": str(row.get("Title") or ""),
                "Evidence": compact_terms(row),
            }
        )
    return pd.DataFrame(rows)


def catalyst_evidence_text(catalyst_board: pd.DataFrame, *, limit: int = 3) -> str:
    if catalyst_board is None or catalyst_board.empty:
        return ""
    rows = []
    for row in catalyst_board.head(limit).to_dict("records"):
        rows.append(
            f"{row.get('Catalyst')}: {row.get('Tone')} "
            f"({row.get('Articles')} articles; {row.get('Evidence')})"
        )
    return "; ".join(rows)


def catalyst_evidence(frame: pd.DataFrame) -> str:
    terms: list[str] = []
    for column in ("Bullish Terms", "Bearish Terms"):
        for value in frame.get(column, pd.Series(dtype=str)).astype(str).tolist():
            terms.extend([term.strip() for term in value.split(",") if term.strip()])
    common = [term for term, _count in Counter(terms).most_common(4)]
    return ", ".join(common) if common else "review latest headlines"


def catalyst_action_read(topic: str, score: float) -> str:
    if score >= 0.15:
        return f"{topic} is supportive; look for confirmation."
    if score <= -0.15:
        return f"{topic} is a risk; require tighter invalidation."
    return f"{topic} is mixed; do not over-weight it."


def nlp_provider_status(
    *,
    fmp_key: str | None = None,
    openai_key: str | None = None,
    x_bearer_token: str | None = None,
    reddit_client_id: str | None = None,
) -> pd.DataFrame:
    """Return provider readiness for the discretionary sentiment cockpit."""

    finbert_available = module_available("transformers") and module_available("torch")
    rows = [
        SentimentProviderStatus(
            provider="Local Lexicon",
            role="Fast offline scoring of FMP/news text",
            status="ready",
            detail="No network or model dependency.",
        ),
        SentimentProviderStatus(
            provider="FMP News/Articles",
            role="Company news/articles source",
            status="configured" if fmp_key else "missing key",
            detail="Used to populate the article cache before local NLP.",
        ),
        SentimentProviderStatus(
            provider="OpenAI Extractor",
            role="Optional article summarization and entity/event extraction",
            status="configured" if openai_key else "optional",
            detail="Useful later for structured catalysts, supply-chain links, and thesis bullets.",
        ),
        SentimentProviderStatus(
            provider="FinBERT",
            role="Optional financial-language sentiment model",
            status="available" if finbert_available else "optional",
            detail="Requires local transformers/torch install before use.",
        ),
        SentimentProviderStatus(
            provider="X / Twitter",
            role="Optional retail sentiment and attention feed",
            status="configured" if x_bearer_token else "not configured",
            detail="Provider hook only; no social API calls are made by this dashboard yet.",
        ),
        SentimentProviderStatus(
            provider="Reddit",
            role="Optional retail discussion feed",
            status="configured" if reddit_client_id else "not configured",
            detail="Provider hook only; no social API calls are made by this dashboard yet.",
        ),
    ]
    return pd.DataFrame([row.as_row() for row in rows])


def score_text_sentiment(text: str) -> tuple[float, list[str], list[str]]:
    clean = normalize_text(text)
    tokens = tokenize(clean)
    bullish = [term for term in tokens if term in BULLISH_TERMS]
    bearish = [term for term in tokens if term in BEARISH_TERMS]
    for phrase in BULLISH_PHRASES:
        if phrase in clean:
            bullish.append(phrase)
    for phrase in BEARISH_PHRASES:
        if phrase in clean:
            bearish.append(phrase)
    score = (len(bullish) - len(bearish)) / max(math.sqrt(len(tokens) or 1), 4.0)
    return clip(score), bullish, bearish


def keyword_summary(counter: Counter[str], *, limit: int = 25) -> pd.DataFrame:
    rows = []
    for keyword, count in counter.most_common(limit):
        tone = "Bullish" if keyword in BULLISH_TERMS else "Bearish" if keyword in BEARISH_TERMS else "Neutral"
        rows.append({"Keyword": keyword, "Count": int(count), "Tone": tone})
    return pd.DataFrame(rows, columns=["Keyword", "Count", "Tone"])


def topic_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["Topic", "Count", "Avg Sentiment"])
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby("Topic", as_index=False)
        .agg(Count=("Score", "size"), **{"Avg Sentiment": ("Score", "mean")})
        .sort_values(["Count", "Avg Sentiment"], ascending=[False, False])
    )
    return grouped.reset_index(drop=True)


def classify_topics(tokens: list[str]) -> list[str]:
    token_set = set(tokens)
    topics = [topic for topic, words in TOPIC_KEYWORDS.items() if token_set.intersection(words)]
    return topics or ["General"]


def split_topics(value: Any) -> list[str]:
    return [topic.strip() for topic in str(value or "").split(",") if topic.strip()]


def compact_terms(row: dict[str, Any]) -> str:
    bullish = str(row.get("Bullish Terms") or "").strip()
    bearish = str(row.get("Bearish Terms") or "").strip()
    pieces = []
    if bullish:
        pieces.append(f"bullish: {bullish}")
    if bearish:
        pieces.append(f"bearish: {bearish}")
    return "; ".join(pieces) if pieces else "headline/topic evidence"


def normalize_dashboard_date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return parsed.date().isoformat()


def module_available(name: str) -> bool:
    return find_spec(name) is not None


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", normalize_text(text))
    return [token for token in tokens if token not in STOP_WORDS]


def sentiment_label(score: float) -> str:
    if score >= 0.20:
        return "Positive"
    if score <= -0.20:
        return "Negative"
    if score >= 0.06:
        return "Slightly Positive"
    if score <= -0.06:
        return "Slightly Negative"
    return "Mixed / Neutral"


def iter_article_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, pd.DataFrame):
        return [dict(row) for row in payload.to_dict("records")]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "articles", "content", "results"):
            nested = payload.get(key)
            if isinstance(nested, list):
                return [row for row in nested if isinstance(row, dict)]
        return [payload]
    return []


def normalize_datetime(value: str) -> str:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return parsed.isoformat()


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def cache_age_hours(value: Any, *, now: datetime | None = None) -> float | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    current = now or datetime.now(timezone.utc)
    return max((current - parsed.to_pydatetime()).total_seconds() / 3600.0, 0.0)


def normalize_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def empty_news_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "published_at", "title", "site", "url", "text", "source", "fetched_at"])


def empty_scored_article_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Published",
            "Title",
            "Site",
            "Sentiment",
            "Score",
            "Topics",
            "Bullish Terms",
            "Bearish Terms",
            "URL",
        ]
    )


def empty_catalyst_board_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Catalyst",
            "Articles",
            "Tone",
            "Avg Score",
            "Latest",
            "Latest Headline",
            "Evidence",
            "Action Read",
        ]
    )
