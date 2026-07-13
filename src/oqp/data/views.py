"""Purpose-specific market-data views."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from oqp.data.brownian_bridge import BrownianBridgeConfig, build_brownian_bridge_view
from oqp.data.missingness import build_accounting_view, build_alpha_view
from oqp.data.quality_flags import summarize_quality


@dataclass(frozen=True)
class MarketDataViews:
    raw: pd.DataFrame
    accounting: pd.DataFrame
    alpha: pd.DataFrame
    risk: pd.DataFrame
    quality_summary: dict[str, Any]


def build_market_data_views(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "date",
    asset_col: str = "ticker",
    price_cols: Sequence[str] = ("close",),
    max_stale_bars: int = 3,
    calendar: Iterable[pd.Timestamp] | None = None,
    risk_imputation: str = "ffill",
    bridge_config: BrownianBridgeConfig | None = None,
) -> MarketDataViews:
    """Build raw/accounting/alpha/risk views from one market-data frame."""

    accounting = build_accounting_view(
        frame,
        timestamp_col=timestamp_col,
        asset_col=asset_col,
        value_cols=price_cols,
        max_stale_bars=max_stale_bars,
        calendar=calendar,
    )
    alpha = build_alpha_view(accounting, value_cols=price_cols)
    risk_mode = risk_imputation.strip().lower()
    if risk_mode in {"ffill", "ffill_with_freshness_flags"}:
        risk = accounting.copy()
        risk.attrs.update(accounting.attrs)
        risk.attrs["view_type"] = "risk"
        risk.attrs["risk_imputation"] = "ffill_with_freshness_flags"
    elif risk_mode == "brownian_bridge":
        if bridge_config is None:
            risk = build_brownian_bridge_view(
                frame,
                timestamp_col=timestamp_col,
                asset_col=asset_col,
                value_cols=price_cols,
                max_gap_bars=max_stale_bars,
                calendar=calendar,
            )
        else:
            risk = build_brownian_bridge_view(
                frame,
                timestamp_col=bridge_config.timestamp_col,
                asset_col=bridge_config.asset_col,
                value_cols=bridge_config.value_cols,
                max_gap_bars=bridge_config.max_gap_bars,
                calendar=calendar,
                seed=bridge_config.seed,
                path_id=bridge_config.path_id,
                sigma_floor=bridge_config.sigma_floor,
            )
    else:
        raise ValueError(
            "risk_imputation must be 'ffill', 'ffill_with_freshness_flags', or 'brownian_bridge'."
        )

    return MarketDataViews(
        raw=frame.copy(),
        accounting=accounting,
        alpha=alpha,
        risk=risk,
        quality_summary=summarize_quality(accounting),
    )
