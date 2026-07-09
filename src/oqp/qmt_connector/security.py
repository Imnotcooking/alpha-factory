"""Security helpers shared by the OQP QMT adapter and connector."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode


SIGNATURE_VERSION = "v1"
SIGNATURE_ALGORITHM = "HMAC-SHA256"
SIGNATURE_HEADER = "X-OQP-Signature"
SIGNATURE_VERSION_HEADER = "X-OQP-Signature-Version"
SIGNATURE_TIMESTAMP_HEADER = "X-OQP-Timestamp"
SIGNATURE_NONCE_HEADER = "X-OQP-Nonce"
DEFAULT_SIGNATURE_TOLERANCE_SECONDS = 30


@dataclass(frozen=True, slots=True)
class SignatureVerificationResult:
    ok: bool
    nonce: str | None = None
    error: str | None = None


def qmt_json_body_bytes(payload: Mapping[str, Any] | None) -> bytes:
    if payload is None:
        return b""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def qmt_signature_headers(
    secret: str,
    method: str,
    path: str,
    *,
    params: Mapping[str, Any] | str | None = None,
    body: bytes = b"",
    timestamp: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    active_timestamp = int(time.time()) if timestamp is None else int(timestamp)
    active_nonce = nonce or uuid.uuid4().hex
    signature = qmt_signature(
        secret,
        method,
        path,
        params=params,
        body=body,
        timestamp=active_timestamp,
        nonce=active_nonce,
    )
    return {
        SIGNATURE_VERSION_HEADER: SIGNATURE_VERSION,
        SIGNATURE_TIMESTAMP_HEADER: str(active_timestamp),
        SIGNATURE_NONCE_HEADER: active_nonce,
        SIGNATURE_HEADER: signature,
    }


def verify_qmt_signature(
    headers: Mapping[str, Any],
    secret: str,
    method: str,
    path: str,
    *,
    params: Mapping[str, Any] | str | None = None,
    body: bytes = b"",
    now: int | None = None,
    tolerance_seconds: int = DEFAULT_SIGNATURE_TOLERANCE_SECONDS,
) -> SignatureVerificationResult:
    version = _header(headers, SIGNATURE_VERSION_HEADER)
    if version != SIGNATURE_VERSION:
        return SignatureVerificationResult(False, error="missing or unsupported signature version")

    timestamp_text = _header(headers, SIGNATURE_TIMESTAMP_HEADER)
    nonce = _header(headers, SIGNATURE_NONCE_HEADER)
    supplied_signature = _header(headers, SIGNATURE_HEADER)
    if not timestamp_text or not nonce or not supplied_signature:
        return SignatureVerificationResult(False, error="missing signature headers")

    try:
        timestamp = int(timestamp_text)
    except ValueError:
        return SignatureVerificationResult(False, error="invalid signature timestamp")

    current = int(time.time()) if now is None else int(now)
    if abs(current - timestamp) > tolerance_seconds:
        return SignatureVerificationResult(False, nonce=nonce, error="stale signature timestamp")

    expected_signature = qmt_signature(
        secret,
        method,
        path,
        params=params,
        body=body,
        timestamp=timestamp,
        nonce=nonce,
    )
    if not hmac.compare_digest(expected_signature, supplied_signature):
        return SignatureVerificationResult(False, nonce=nonce, error="invalid signature")
    return SignatureVerificationResult(True, nonce=nonce)


def qmt_signature(
    secret: str,
    method: str,
    path: str,
    *,
    params: Mapping[str, Any] | str | None = None,
    body: bytes = b"",
    timestamp: int,
    nonce: str,
) -> str:
    message = _canonical_message(
        method,
        path,
        params=params,
        body=body,
        timestamp=timestamp,
        nonce=nonce,
    )
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def canonical_qmt_query(params: Mapping[str, Any] | str | None) -> str:
    pairs: list[tuple[str, str]] = []
    if params is None:
        return ""
    if isinstance(params, str):
        raw_pairs = parse_qsl(params, keep_blank_values=True)
        pairs.extend((str(key), str(value)) for key, value in raw_pairs)
    else:
        for key, value in params.items():
            if value is None:
                continue
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                pairs.extend((str(key), str(item)) for item in value)
            else:
                pairs.append((str(key), str(value)))
    pairs.sort(key=lambda pair: (pair[0], pair[1]))
    return urlencode(pairs)


def _canonical_message(
    method: str,
    path: str,
    *,
    params: Mapping[str, Any] | str | None,
    body: bytes,
    timestamp: int,
    nonce: str,
) -> bytes:
    body_digest = hashlib.sha256(body or b"").hexdigest()
    lines = (
        method.upper(),
        path or "/",
        canonical_qmt_query(params),
        body_digest,
        str(timestamp),
        nonce,
    )
    return "\n".join(lines).encode("utf-8")


def _header(headers: Mapping[str, Any], name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        lower_name = name.lower()
        for key, candidate in headers.items():
            if str(key).lower() == lower_name:
                value = candidate
                break
    if value is None:
        return None
    text = str(value).strip()
    return text or None
