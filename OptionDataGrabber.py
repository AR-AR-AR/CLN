#!/usr/bin/env python3
"""
gex_snapshot_loop_v3.py — Standalone (no CLI args, no Dash).
- Polls Tastytrade every 5 minutes
- Computes GEX/DEX/Charm and overwrites latest CSV
- ALSO writes a second CSV with ONLY the top 5 highest positive GEX levels
  in the requested "#automap BMD" format; all fields fixed except Price Level & Note.

Requirements:
  pip install tastytrade pandas numpy

Run:
  python gex_snapshot_loop_v3.py
Press Ctrl+C to stop.
"""

import sys
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tastytrade import Session, DXLinkStreamer
from tastytrade.dxfeed import Greeks, Summary
from tastytrade.market_data import get_market_data
from tastytrade.utils import today_in_new_york, is_market_open_on
from tastytrade.instruments import NestedOptionChain
from tastytrade.order import InstrumentType

# ── HARD-CODED SETTINGS ──────────────────────────────────────────────────────
EMAIL = "araseley"          # <-- put your Tastytrade email here
PASSWORD = "Not_4_you91!"    # <-- put your Tastytrade password here

SYMBOL = "SPX"           # Underlying symbol (e.g., SPX, SPY, ESU5)
STRIKE_RANGE = 120.0     # +/- strikes around spot
DTE_LIMIT = 1            # Include expirations with DTE <= this
TIMEOUT_SECONDS = 15     # Stream snapshot timeout
OUTDIR = "./csv"         # Output directory

WRITE_DAILY = False      # Append daily file (SYMBOL_gex_YYYYMMDD.csv)
WRITE_LATEST = True      # Overwrite latest file (SYMBOL_gex_latest.csv)

# Top-5 automap output config
AUTOMAP_FILENAME = None  # If None, defaults to f"{SYMBOL.upper()}_top5_gex.csv"
# Fixed fields (kept same for all rows; only Price Level & Note vary)
AUTOMAP_FOREGROUND = "#ffffff"
AUTOMAP_BACKGROUND = "#FF0000"
AUTOMAP_TEXT_ALIGN = "center"
AUTOMAP_DIAMETER = 2
AUTOMAP_DRAW_HLINE = "TRUE"

LOOP_INTERVAL_SEC = 300  # 5 minutes
# ─────────────────────────────────────────────────────────────────────────────


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


async def _collect_greeks(streamer: DXLinkStreamer, greeks_out: Dict[str, Dict], subs: List[str], timeout: int):
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
            # Let timeout govern; no forced break (reduces generator exit warnings)
    await asyncio.wait_for(_listen(), timeout=timeout)


async def _collect_summary(streamer: DXLinkStreamer, summary_out: Dict[str, Dict], subs: List[str], timeout: int):
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
            # Let timeout govern; no forced break
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
                    await _collect_greeks(streamer, greeks_data, subs, timeout)
                except asyncio.TimeoutError:
                    pass

            async def run_s():
                try:
                    await _collect_summary(streamer, summary_data, subs, timeout)
                except asyncio.TimeoutError:
                    pass

            await asyncio.gather(run_g(), run_s())

    try:
        asyncio.run(_runner())
    except RuntimeError:
        # Fallback if already inside an event loop
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


def save_top5_automap(gex_df: pd.DataFrame, symbol: str, outdir: Path) -> Path:
    """
    Save ONLY the top 5 highest positive Net_GEX levels into a CSV with this format:

    First line:
        #automap BMD
    Then header row:
        Symbol,Price Level,Note,Foreground Color,Background Color,Text Alignment,DIAMETER, DRAW_NOTE_PRICE_HORIZONTAL_LINE
    Then rows (fixed fields except Price Level and Note). Note includes the price and GEX value.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    fname = AUTOMAP_FILENAME or f"{symbol.upper()}_top5_gex.csv"
    path = outdir / fname

    pos = gex_df[gex_df["Net_GEX"] > 0].copy()
    if pos.empty:
        with open(path, "w", newline="") as f:
            f.write("#automap BMD\n")
            f.write("Symbol,Price Level,Note,Foreground Color,Background Color,Text Alignment,DIAMETER, DRAW_NOTE_PRICE_HORIZONTAL_LINE\n")
        return path

    top5 = pos.sort_values("Net_GEX", ascending=False).head(5).copy()

    rows = []
    for _, r in top5.iterrows():
        strike = float(r["strike"])
        gex_val = float(r["Net_GEX"])
        note = f"Price {strike:.0f} | GEX {gex_val:,.0f}"
        rows.append({
            "Symbol": symbol.upper(),
            "Price Level": f"{strike:.0f}",  # keep as integer-like string
            "Note": note,
            "Foreground Color": AUTOMAP_FOREGROUND,
            "Background Color": AUTOMAP_BACKGROUND,
            "Text Alignment": AUTOMAP_TEXT_ALIGN,
            "DIAMETER": AUTOMAP_DIAMETER,
            " DRAW_NOTE_PRICE_HORIZONTAL_LINE": AUTOMAP_DRAW_HLINE,
        })

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

    # Write with the required first line
    with open(path, "w", newline="") as f:
        f.write("#automap BMD\n")
        out_df.to_csv(f, index=False)

    return path


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
        spot, vix_data = fetch_spot_and_vix(session, SYMBOL)
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] Failed to fetch spot/VIX: {e}", file=sys.stderr, flush=True)
        return False

    try:
        subs, opt_df = build_option_subs(session, SYMBOL, spot, STRIKE_RANGE, DTE_LIMIT)
        if not subs:
            print(f"[{datetime.now():%H:%M:%S}] No option symbols matched filters.", file=sys.stderr, flush=True)
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

    # 1) Save the main snapshot(s)
    written_paths = save_main_csv(gex_df, spot, vix_data, SYMBOL, outdir, WRITE_LATEST or not WRITE_DAILY, WRITE_DAILY)
    if not written_paths:
        print(f"[{datetime.now():%H:%M:%S}] Nothing written (main CSV).", file=sys.stderr, flush=True)
        return False

    # 2) Save the top-5 automap file
    top5_path = save_top5_automap(gex_df, SYMBOL, outdir)

    flip_idx = gex_df["Cumulative_GEX"].abs().idxmin()
    flip_strike = float(gex_df.loc[flip_idx, "strike"])
    print(
        f"[{datetime.now():%H:%M:%S}] Wrote {len(written_paths)} main file(s) "
        f"+ top-5 automap at {top5_path.resolve()}. "
        f"Spot={spot:.2f} VIX={vix_data['current']:.2f} Flip={flip_strike:.2f}",
        flush=True
    )
    for p in written_paths:
        print(f"  -> {p.resolve()}", flush=True)
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
        print("ERROR: Edit EMAIL/PASSWORD at the top of gex_snapshot_loop_v3.py before running.", file=sys.stderr)
        sys.exit(2)
    main_loop()
