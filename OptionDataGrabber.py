#!/usr/bin/env python3
"""
gex_snapshot_loop_v5.py — Standalone (no CLI args, no Dash).

- Polls Tastytrade every 5 minutes
- Computes GEX/DEX/Charm on SPX and overwrites the main CSV
- Builds an "automap" CSV with:
    * Top 5 positive Net_GEX levels labeled GEX1..GEX5 (highest first)
    * CALL_WALL (max Call_OI), PUT_WALL (max Put_OI), MAX_PAIN (min OI payout)
  but prices are mapped to the current ES contract via:
      ES_level = SPX_level + (ES_last - SPX_spot)

- The automap CSV "Symbol" column uses "<current ES contract>.CME@BMD" (e.g., ESU5.CME@BMD)
- After writing files, automatically runs: git add → commit → push

Requirements:
  pip install tastytrade pandas numpy
Run:
  python gex_snapshot_loop_v5.py
"""

import sys
import time
import asyncio
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from tastytrade import Session, DXLinkStreamer
from tastytrade.dxfeed import Greeks, Summary
from tastytrade.market_data import get_market_data, get_market_data_by_type
from tastytrade.utils import today_in_new_york, is_market_open_on
from tastytrade.instruments import NestedOptionChain
from tastytrade.order import InstrumentType

# ── HARD-CODED SETTINGS ──────────────────────────────────────────────────────
EMAIL = "araseley"          # <-- put your Tastytrade email here
PASSWORD = "Not_4_you91!"    # <-- put your Tastytrade password here

SYMBOL = "SPX"           # Underlying for greeks (SPX index)
STRIKE_RANGE = 120.0     # +/- strikes around spot
DTE_LIMIT = 1            # Include expirations with DTE <= this
TIMEOUT_SECONDS = 15     # Stream snapshot timeout
OUTDIR = "./csv"         # Output directory

WRITE_DAILY = False      # Append daily file (SYMBOL_gex_YYYYMMDD.csv)
WRITE_LATEST = True      # Overwrite latest file (SYMBOL_gex_latest.csv)

# Automap output config
AUTOMAP_FILENAME = None  # If None, defaults to f"{ES_CONTRACT}.CME@BMD_top5_gex.csv"
AUTOMAP_FOREGROUND = "#ffffff"
AUTOMAP_BACKGROUND = "#FF0000"
AUTOMAP_TEXT_ALIGN = "center"
AUTOMAP_DIAMETER = 2
AUTOMAP_DRAW_HLINE = "TRUE"

# ES contract override (optional). If None, script auto-detects front contract code.
ES_CONTRACT_OVERRIDE: Optional[str] = "ESU5"   # e.g., "ESU5"

LOOP_INTERVAL_SEC = 300  # 5 minutes
GIT_BRANCH = "main"      # branch to push
# ─────────────────────────────────────────────────────────────────────────────


def current_es_contract_symbol(dt: datetime) -> str:
    """Return the current ES (E-mini S&P 500) quarterly contract code like 'ESU5'."""
    # CME quarter codes: Mar=H, Jun=M, Sep=U, Dec=Z
    month = dt.month
    year_last = dt.year % 10
    if month <= 3:
        code = "H"
    elif month <= 6:
        code = "M"
    elif month <= 9:
        code = "U"
    else:
        code = "Z"
    return f"ES{code}{year_last}"


def fetch_spot_and_vix(session: Session, symbol: str) -> Tuple[float, Dict[str, float]]:
    mdata = get_market_data(session, symbol, InstrumentType.INDEX)
    spot_price = float(mdata.last)

    # VIX with safe fallbacks
    try:
        vix_mdata = get_market_data(session, "VIX", InstrumentType.INDEX)
        vix_current = float(vix_mdata.last)
    except Exception:
        vix_current = 20.0
    try:
        vix9d_mdata = get_market_data(session, "VIX9D", InstrumentType.INDEX)
        vix9d = float(vix9d_mdata.last)
    except Exception:
        vix9d = vix_current * 0.95
    try:
        vix3m_mdata = get_market_data(session, "VIX3M", InstrumentType.INDEX)
        vix3m = float(vix3m_mdata.last)
    except Exception:
        vix3m = vix_current * 1.05

    vix_data = {
        "current": vix_current,
        "vix9d": vix9d,
        "vix3m": vix3m,
        "term_structure": (vix3m / vix_current) if vix_current else 1.0,
        "short_term_premium": (vix_current / vix9d) if vix9d else 1.0,
    }
    return spot_price, vix_data


def fetch_es_price(session: Session, override: Optional[str] = None) -> Tuple[str, float]:
    """
    Return (es_contract_code, last_price) for ES.
    If override is provided, use it; else derive from today's date.
    """
    contract = override or current_es_contract_symbol(datetime.now())
    # Tastytrade futures symbol is typically like 'ESU5'
    m = get_market_data_by_type(session, futures=["/"+contract])
    #m = get_market_data(session, contract, InstrumentType.FUTURE)
    return contract, float(m[0].last)


def build_option_subs(session: Session, symbol: str, spot: float, strike_range: float, dte_limit: int) -> Tuple[List[str], pd.DataFrame]:
    chain = NestedOptionChain.get(session, symbol)

    subs_list: List[str] = []
    rows: List[Dict] = []

    for exp in chain[0].expirations:
        if exp.days_to_expiration <= dte_limit:
            for strike in exp.strikes:
                k = float(strike.strike_price)
                if (spot - strike_range) <= k <= (spot + strike_range):
                    subs_list.append(strike.call_streamer_symbol)
                    subs_list.append(strike.put_streamer_symbol)
                    rows.append(
                        {
                            "strike": k,
                            "call_symbol": strike.call_streamer_symbol,
                            "put_symbol": strike.put_streamer_symbol,
                            "expiration": exp.expiration_date,
                            "dte": exp.days_to_expiration,
                        }
                    )
    return subs_list, pd.DataFrame(rows)


async def _collect_greeks(streamer: DXLinkStreamer, greeks_out: Dict[str, Dict], timeout: int):
    async def _listen():
        async for event in streamer.listen(Greeks):
            greeks_out[event.event_symbol] = {
                "symbol": event.event_symbol,
                "delta": float(event.delta or 0.0),
                "gamma": float(event.gamma or 0.0),
                "vega": float(event.vega or 0.0),
                "theta": float(event.theta or 0.0),
                "iv": float(event.volatility or 0.0),
            }
    await asyncio.wait_for(_listen(), timeout=timeout)


async def _collect_summary(streamer: DXLinkStreamer, summary_out: Dict[str, Dict], timeout: int):
    async def _listen():
        async for event in streamer.listen(Summary):
            oi = event.open_interest
            try:
                oi_val = float(oi) if oi is not None else 0.0
            except Exception:
                oi_val = 0.0
            summary_out[event.event_symbol] = {
                "symbol": event.event_symbol,
                "open_interest": oi_val,
            }
    await asyncio.wait_for(_listen(), timeout=timeout)


def stream_snapshot(session: Session, subs: List[str], timeout: int) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    greeks_data: Dict[str, Dict] = {}
    summary_data: Dict[str, Dict] = {}

    async def _runner():
        async with DXLinkStreamer(session) as streamer:
            await streamer.subscribe(Greeks, subs)
            await streamer.subscribe(Summary, subs)
            async def run_g():
                try:
                    await _collect_greeks(streamer, greeks_data, timeout)
                except asyncio.TimeoutError:
                    pass
            async def run_s():
                try:
                    await _collect_summary(streamer, summary_data, timeout)
                except asyncio.TimeoutError:
                    pass
            await asyncio.gather(run_g(), run_s())

    try:
        asyncio.run(_runner())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_runner())
        finally:
            loop.close()

    return greeks_data, summary_data


def process_exposures(greeks: Dict[str, Dict], summary: Dict[str, Dict], options_df: pd.DataFrame, dte_limit: int) -> pd.DataFrame:
    if not greeks or options_df.empty:
        return pd.DataFrame()

    df_g = pd.DataFrame(greeks.values())
    df_o = options_df.copy()

    if summary:
        df_s = pd.DataFrame(summary.values())
    else:
        df_s = pd.DataFrame(columns=["symbol", "open_interest"])

    # Normalize types
    for col in ["delta", "gamma", "vega", "theta", "iv"]:
        if col in df_g.columns:
            df_g[col] = pd.to_numeric(df_g[col], errors="coerce").fillna(0.0).astype(float)
    if "open_interest" not in df_s.columns:
        df_s["open_interest"] = 0.0
    else:
        df_s["open_interest"] = pd.to_numeric(df_s["open_interest"], errors="coerce").fillna(0.0).astype(float)
    df_o["strike"] = pd.to_numeric(df_o["strike"], errors="coerce").astype(float)

    # Merge Summary into Greeks first so OI is always present
    df_gs = df_g.merge(df_s[["symbol", "open_interest"]], on="symbol", how="left").fillna({"open_interest": 0.0})

    # Calls
    calls = df_gs.merge(df_o, left_on="symbol", right_on="call_symbol", how="inner")
    calls["Call_GEX"] = calls["gamma"] * calls["open_interest"] * 100.0
    calls["Call_DEX"] = calls["delta"] * calls["open_interest"] * 100.0
    calls["Call_IV"] = calls["iv"]
    calls_grp = (
        calls.groupby("strike")
        .agg({"Call_GEX": "sum", "Call_DEX": "sum", "open_interest": "sum", "Call_IV": "mean"})
        .rename(columns={"open_interest": "Call_OI"})
        .reset_index()
    )

    # Puts
    puts = df_gs.merge(df_o, left_on="symbol", right_on="put_symbol", how="inner")
    puts["Put_GEX"] = puts["gamma"] * puts["open_interest"] * 100.0
    puts["Put_DEX"] = puts["delta"] * puts["open_interest"] * 100.0
    puts["Put_IV"] = puts["iv"]
    puts_grp = (
        puts.groupby("strike")
        .agg({"Put_GEX": "sum", "Put_DEX": "sum", "open_interest": "sum", "Put_IV": "mean"})
        .rename(columns={"open_interest": "Put_OI"})
        .reset_index()
    )

    gex_df = calls_grp.merge(puts_grp, on="strike", how="outer").fillna(0.0)
    gex_df["Net_GEX"] = gex_df["Call_GEX"] - gex_df["Put_GEX"]
    gex_df["Net_DEX"] = gex_df["Call_DEX"] + gex_df["Put_DEX"]
    gex_df["Charm"] = gex_df["Net_DEX"].diff().fillna(0.0) / max(1, dte_limit)
    gex_df["Cumulative_GEX"] = gex_df["Net_GEX"].cumsum()
    gex_df["Total_OI"] = gex_df["Call_OI"] + gex_df["Put_OI"]

    return gex_df.sort_values("strike").reset_index(drop=True)


def save_main_csv(gex_df: pd.DataFrame, spot: float, vix: Dict[str, float], symbol: str, outdir: Path, write_latest: bool, write_daily: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = gex_df.copy()
    if df.empty:
        return []

    # Prepend metadata columns
    df.insert(0, "timestamp", ts)
    df.insert(1, "symbol", symbol.upper())
    df.insert(2, "spot", float(spot))
    df.insert(3, "vix", float(vix.get("current", 0.0)))

    written = []
    if write_latest:
        latest_path = outdir / f"{symbol.upper()}_gex_latest.csv"
        df.to_csv(latest_path, index=False)
        written.append(latest_path)
    if write_daily:
        date_tag = datetime.now().strftime("%Y%m%d")
        daily_path = outdir / f"{symbol.upper()}_gex_{date_tag}.csv"
        header_needed = not daily_path.exists()
        df.to_csv(daily_path, mode="a", index=False, header=header_needed)
        written.append(daily_path)
    return written


def call_wall_put_wall_from_oi(gex_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Return (call_wall_strike, put_wall_strike) using max Call_OI and max Put_OI (fallback to GEX if OI missing)."""
    call_wall = None
    put_wall = None
    if "Call_OI" in gex_df.columns and gex_df["Call_OI"].sum() > 0:
        call_wall = float(gex_df.loc[gex_df["Call_OI"].idxmax(), "strike"])
    elif "Call_GEX" in gex_df.columns and gex_df["Call_GEX"].abs().sum() > 0:
        call_wall = float(gex_df.loc[gex_df["Call_GEX"].idxmax(), "strike"])
    if "Put_OI" in gex_df.columns and gex_df["Put_OI"].sum() > 0:
        put_wall = float(gex_df.loc[gex_df["Put_OI"].idxmax(), "strike"])
    elif "Put_GEX" in gex_df.columns and gex_df["Put_GEX"].abs().sum() > 0:
        put_wall = float(gex_df.loc[gex_df["Put_GEX"].idxmax(), "strike"])
    return call_wall, put_wall


def max_pain_from_oi(gex_df: pd.DataFrame) -> Optional[float]:
    """
    Max Pain on strike grid by minimizing OI payout:
      payout(P) = sum(Call_OI_i * max(0, P - K_i)) + sum(Put_OI_i * max(0, K_i - P))
    Returns strike P with minimal payout or None.
    """
    need = {"strike", "Call_OI", "Put_OI"}
    if not need.issubset(gex_df.columns):
        return None
    strikes = gex_df["strike"].to_numpy(dtype=float)
    call_oi = gex_df["Call_OI"].to_numpy(dtype=float)
    put_oi = gex_df["Put_OI"].to_numpy(dtype=float)
    if (np.nansum(call_oi) == 0) and (np.nansum(put_oi) == 0):
        return None
    payouts = []
    for P in strikes:
        call_pay = np.sum(call_oi * np.maximum(0.0, P - strikes))
        put_pay = np.sum(put_oi * np.maximum(0.0, strikes - P))
        payouts.append(call_pay + put_pay)
    idx = int(np.nanargmin(payouts))
    return float(strikes[idx])


def save_top5_automap(
    gex_df: pd.DataFrame,
    es_contract: str,
    es_offset: float,
    outdir: Path
) -> Path:
    """
    Save top 5 positive Net_GEX + CALL_WALL / PUT_WALL / MAX_PAIN mapped to ES via `es_offset`.
    Price Level written as: round(SPX_level + es_offset) with 0 decimals.
    Symbol: f"{es_contract}.CME@BMD"
    """
    outdir.mkdir(parents=True, exist_ok=True)
    display_symbol = f"{es_contract}.CME@BMD"
    fname = AUTOMAP_FILENAME or f"SPXES_top5_gex.csv"
    path = outdir / fname

    rows: List[Dict] = []

    # Top 5 positive Net_GEX (highest first)
    pos = gex_df[gex_df["Net_GEX"] > 0].copy()
    pos = pos.sort_values("Net_GEX", ascending=False).head(5)
    for i, (_, r) in enumerate(pos.iterrows(), start=1):
        spx_level = float(r["strike"])
        es_level = spx_level + es_offset
        rows.append({
            "Symbol": display_symbol,
            "Price Level": f"{es_level:.0f}",
            "Note": f"GEX{i}",
            "Foreground Color": AUTOMAP_FOREGROUND,
            "Background Color": AUTOMAP_BACKGROUND,
            "Text Alignment": AUTOMAP_TEXT_ALIGN,
            "DIAMETER": AUTOMAP_DIAMETER,
            " DRAW_NOTE_PRICE_HORIZONTAL_LINE": AUTOMAP_DRAW_HLINE,
        })

    # Walls & Max Pain (compute on SPX grid, then map to ES)
    cw, pw = call_wall_put_wall_from_oi(gex_df)
    mp = max_pain_from_oi(gex_df)

    def add_row_if_finite(spx_price: Optional[float], label: str):
        if spx_price is not None and np.isfinite(spx_price):
            es_price = float(spx_price) + es_offset
            rows.append({
                "Symbol": display_symbol,
                "Price Level": f"{es_price:.0f}",
                "Note": label,
                "Foreground Color": AUTOMAP_FOREGROUND,
                "Background Color": AUTOMAP_BACKGROUND,
                "Text Alignment": AUTOMAP_TEXT_ALIGN,
                "DIAMETER": AUTOMAP_DIAMETER,
                "DRAW_NOTE_PRICE_HORIZONTAL_LINE": AUTOMAP_DRAW_HLINE,
            })

    add_row_if_finite(cw, "CALL_WALL")
    add_row_if_finite(pw, "PUT_WALL")
    add_row_if_finite(mp, "MAX_PAIN")

    out_df = pd.DataFrame(rows, columns=[
        "Symbol",
        "Price Level",
        "Note",
        "Foreground Color",
        "Background Color",
        "Text Alignment",
        "DIAMETER",
        " DRAW_NOTE_PRICE_HORIZONTAL_LINE",
    ])

    with open(path, "w", newline="") as f:
        #f.write("#automap BMD\n")
        out_df.to_csv(f, index=False)

    return path


def git_add_commit_push(paths: List[Path], message: Optional[str] = None, branch: str = GIT_BRANCH) -> None:
    """git add/commit/push in the repo containing this script."""
    if not paths:
        return
    repo_dir = Path(__file__).resolve().parent
    # Convert to relative paths (nicer in git)
    rels = []
    for p in paths:
        try:
            rels.append(str(Path(p).resolve().relative_to(repo_dir)))
        except Exception:
            rels.append(str(p))

    commit_msg = message or f"Update snapshots {datetime.now():%Y-%m-%d %H:%M:%S}"
    try:
        subprocess.run(["git", "add", *rels], cwd=repo_dir, check=True)
        commit_proc = subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_dir)
        if commit_proc.returncode != 0:
            # likely nothing to commit
            return
        subprocess.run(["git", "push", "origin", branch], cwd=repo_dir, check=True)
    except Exception as e:
        # Non-fatal: print and continue
        print(f"[git] push skipped: {e}", file=sys.stderr, flush=True)


def one_cycle() -> bool:
    _next_exp = today_in_new_york()
    while not is_market_open_on(_next_exp):
        _next_exp += timedelta(days=1)

    try:
        session = Session(EMAIL, PASSWORD)
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] Login failed: {e}", file=sys.stderr, flush=True)
        return False

    try:
        spx_spot, vix_data = fetch_spot_and_vix(session, SYMBOL)
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] Failed to fetch SPX/VIX: {e}", file=sys.stderr, flush=True)
        return False

    try:
        es_contract, es_last = fetch_es_price(session, ES_CONTRACT_OVERRIDE)
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] Failed to fetch ES price: {e}", file=sys.stderr, flush=True)
        return False

    es_offset = es_last - spx_spot  # map SPX levels to ES via additive offset

    try:
        subs, opt_df = build_option_subs(session, SYMBOL, spx_spot, STRIKE_RANGE, DTE_LIMIT)
        if not subs:
            print(f"[{datetime.now():%H:%M:%S}] No SPX option symbols matched filters.", file=sys.stderr, flush=True)
            return False
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] Failed to build option subs: {e}", file=sys.stderr, flush=True)
        return False

    greeks, summary = stream_snapshot(session, subs, TIMEOUT_SECONDS)
    gex_df = process_exposures(greeks, summary, opt_df, DTE_LIMIT)
    if gex_df.empty:
        print(f"[{datetime.now():%H:%M:%S}] No exposures computed.", file=sys.stderr, flush=True)
        return False

    outdir = Path(OUTDIR)

    # 1) Save the main SPX snapshot(s)
    main_paths = save_main_csv(gex_df, spx_spot, vix_data, SYMBOL, outdir, WRITE_LATEST or not WRITE_DAILY, WRITE_DAILY)

    # 2) Save the automap file (mapped to ES) and git push both
    automap_path = save_top5_automap(gex_df, es_contract, es_offset, outdir)

    flip_idx = gex_df["Cumulative_GEX"].abs().idxmin()
    flip_strike = float(gex_df.loc[flip_idx, "strike"])

    print(
        f"[{datetime.now():%H:%M:%S}] SPX spot={spx_spot:.2f} ES({es_contract})={es_last:.2f} "
        f"offset={es_offset:+.2f} | Flip={flip_strike:.0f}",
        flush=True
    )
    for p in main_paths:
        print(f"  -> wrote {p.resolve()}", flush=True)
    print(f"  -> wrote {automap_path.resolve()}", flush=True)

    # Git add/commit/push
    git_add_commit_push([*main_paths, automap_path], message=f"{es_contract} automap + SPX GEX {datetime.now():%Y-%m-%d %H:%M:%S}")

    return True


def main_loop():
    print("Starting GEX snapshot loop (5-minute cadence). Press Ctrl+C to stop.", flush=True)
    print(f"  Underlying: {SYMBOL} | ±Range: {STRIKE_RANGE} | DTE≤{DTE_LIMIT} | outdir: {OUTDIR}", flush=True)
    while True:
        t0 = time.time()
        try:
            one_cycle()
        except KeyboardInterrupt:
            print("\nStopping...", flush=True)
            break
        except Exception as e:
            print(f"[{datetime.now():%H:%M:%S}] Unhandled error: {e}", file=sys.stderr, flush=True)
        elapsed = time.time() - t0
        wait = max(0.0, LOOP_INTERVAL_SEC - elapsed)
        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            print("\nStopping...", flush=True)
            break


if __name__ == "__main__":
    if "YOUR_TASTY_EMAIL" in EMAIL or "YOUR_TASTY_PASSWORD" in PASSWORD:
        print("ERROR: Edit EMAIL/PASSWORD at the top of gex_snapshot_loop_v5.py before running.", file=sys.stderr)
        sys.exit(2)
    main_loop()
