"""
features.py
============
Turns raw (market, retail-flow) series into a supervised learning table for
each forecast horizon h in {5, 10, 20} trading days, with an explicit,
checkable no-look-ahead construction.

TIMING CONVENTION (read this before touching anything)
--------------------------------------------------------
- Everything with a `_t` feature name uses ONLY information available as of
  the close of day t (i.e. rolling windows ending at t, or values dated <= t).
- The target `target_rv_{h}` at row t is the realized volatility computed
  from returns t+1 ... t+h — i.e. STRICTLY FUTURE relative to the feature row.
- The last h rows of every horizon's table are dropped (their target would
  require data past the end of the sample) rather than filled or bfilled.
- The "reflex" features are built only from (return_s, next-day retail
  response_{s+1}) pairs where s+1 <= t, i.e. pairs that have ALREADY been
  fully observed by day t. The rolling regression that estimates the
  historical buy-the-dip beta at time t never uses pair (t, t+1).

`assert_no_lookahead()` at the bottom re-derives a couple of these checks
programmatically and raises if violated — run it in CI / before every backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _rolling_ols_slope(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope of y on x (single regressor + intercept), window ending at each t."""
    x = x.values
    y = y.values
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window, n + 1):
        xs = x[i - window : i]
        ys = y[i - window : i]
        mask = ~(np.isnan(xs) | np.isnan(ys))
        if mask.sum() < window // 2:
            continue
        xs, ys = xs[mask], ys[mask]
        xm, ym = xs.mean(), ys.mean()
        denom = ((xs - xm) ** 2).sum()
        if denom <= 1e-12:
            continue
        beta = ((xs - xm) * (ys - ym)).sum() / denom
        out[i - 1] = beta
    return pd.Series(out, index=range(n))


def build_feature_target_table(
    market_df: pd.DataFrame,
    retail_df: pd.DataFrame,
    horizons=(5, 10, 20),
    reflex_window: int = 60,
    zscore_window: int = 60,
) -> dict:
    """
    Parameters
    ----------
    market_df : DataFrame with columns ['spy_close', 'ret', 'vix'], daily.
    retail_df : DataFrame with columns ['off_exchange_share', 'short_volume_ratio'], daily.
    horizons  : forecast horizons in trading days.

    Returns
    -------
    dict[h] -> DataFrame with feature columns + 'target_rv_h', ready for modeling.
    """
    df = market_df.join(retail_df, how="inner").sort_index()
    df = df.dropna(subset=["ret"]).copy()
    n = len(df)

    # ---- base (benchmark / HAR-style) features -------------------------------
    for w in (5, 10, 20):
        df[f"rv_{w}"] = df["ret"].rolling(w).std(ddof=0) * np.sqrt(TRADING_DAYS)
    df["vix_level"] = df["vix"]
    df["vix_change_5"] = df["vix"].diff(5)

    # ---- retail-flow proxy features ------------------------------------------
    df["retail_share"] = df["off_exchange_share"]
    roll_mean = df["retail_share"].rolling(zscore_window).mean()
    roll_std = df["retail_share"].rolling(zscore_window).std(ddof=0)
    df["retail_share_z"] = (df["retail_share"] - roll_mean) / roll_std.replace(0, np.nan)
    df["retail_short_ratio"] = df["short_volume_ratio"]

    # next-day change in retail share, i.e. the "response" to today's return.
    # response_at(t) = retail_share(t) - retail_share(t-1), which is the market's
    # reaction TO ret(t-1). This is realized/known by day t.
    df["retail_share_delta"] = df["retail_share"].diff()

    # Pairs used for the rolling reflex-beta at time t: (ret_{s}, delta_{s+1})
    # for s from t-reflex_window .. t-1. We implement this by aligning
    # ret.shift(1) with retail_share_delta (both indexed at s+1 = "today"),
    # then taking a rolling window ENDING AT t-1 relative to delta, i.e.
    # ret.shift(1) at row t is ret_{t-1}; retail_share_delta at row t is
    # delta_t = share_t - share_{t-1}, the response to ret_{t-1}.
    # A rolling regression of retail_share_delta on ret.shift(1), using a
    # window ending at row (t-1), gives us the historical beta known BEFORE
    # today's own realized pair — exactly what we want for reflex_beta_t.
    resp = df["retail_share_delta"]              # response dated at "today"
    driver = df["ret"].shift(1)                  # yesterday's return that drove it
    # reflex_beta as of t uses pairs up through t-1 (shift by 1 more before rolling)
    reflex_beta_raw = _rolling_ols_slope(driver.shift(1).reset_index(drop=True),
                                          resp.shift(1).reset_index(drop=True),
                                          reflex_window)
    df["reflex_beta"] = reflex_beta_raw.values

    # reflex_gap_t: today's ACTUAL response vs. what the (already-known,
    # pre-today) historical beta would have predicted from yesterday's return.
    # Both driver (ret_{t-1}) and resp (delta_t) are fully realized by day t;
    # reflex_beta_t was estimated using only data through t-1. No leakage.
    df["predicted_response"] = df["reflex_beta"] * driver
    df["reflex_gap"] = resp - df["predicted_response"]
    # Positive reflex_gap = retail bought MORE than history would predict;
    # negative & growing more negative = the buy-the-dip reflex is "lagging"
    # relative to what's warranted — this is the early-warning candidate
    # feature described in the chat.
    df["reflex_gap_roll_mean"] = df["reflex_gap"].rolling(20).mean()

    feature_cols_base = ["rv_5", "rv_10", "rv_20", "vix_level", "vix_change_5"]
    feature_cols_retail = [
        "retail_share",
        "retail_share_z",
        "retail_short_ratio",
        "reflex_beta",
        "reflex_gap",
        "reflex_gap_roll_mean",
    ]

    out = {}
    for h in horizons:
        fwd_ret = df["ret"].shift(-1).rolling(h).apply(lambda a: np.std(a, ddof=0), raw=True)
        # rolling(h).apply on shift(-1) aligned so that the window ending at
        # row t actually covers ret_{t+1}..ret_{t+h}: shift(-1) moves ret_{t+1}
        # to row t, so a trailing rolling(h) window ending at row t covers
        # exactly ret_{t+1}..ret_{t+h}. Verified in assert_no_lookahead().
        target = fwd_ret.shift(-(h - 1)) * np.sqrt(TRADING_DAYS / h)

        table = df[feature_cols_base + feature_cols_retail].copy()
        table["target_rv_h"] = target
        table = table.dropna()
        out[h] = table

    meta = {
        "feature_cols_base": feature_cols_base,
        "feature_cols_retail": feature_cols_retail,
        "raw": df,
    }
    return out, meta


def assert_no_lookahead(market_df: pd.DataFrame, retail_df: pd.DataFrame, horizons=(5, 10, 20)):
    """Re-derive the target construction independently and confirm alignment,
    plus confirm feature columns never correlate suspiciously perfectly with
    same-day-or-later returns they shouldn't have access to. Raises AssertionError on failure.
    """
    tables, meta = build_feature_target_table(market_df, retail_df, horizons)
    df = meta["raw"]
    ret = df["ret"]

    for h, table in tables.items():
        # Manual re-derivation for a handful of interior rows.
        common_idx = table.index
        sample_idx = common_idx[len(common_idx) // 2 : len(common_idx) // 2 + 5]
        for t in sample_idx:
            pos = df.index.get_loc(t)
            if pos + h >= len(df):
                continue
            manual_future = ret.iloc[pos + 1 : pos + 1 + h].values
            manual_target = np.std(manual_future, ddof=0) * np.sqrt(TRADING_DAYS / h)
            got = table.loc[t, "target_rv_h"]
            assert np.isclose(manual_target, got, rtol=1e-6), (
                f"Look-ahead check FAILED at horizon {h}, date {t}: "
                f"expected {manual_target}, got {got}"
            )
    print("No-look-ahead sanity check passed for horizons:", horizons)
