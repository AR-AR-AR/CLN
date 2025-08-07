"""
Market data fetching and streaming functionality for SPX Trading Dashboard
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta, datetime

from tastytrade import Session, DXLinkStreamer
from tastytrade.dxfeed import Greeks, Summary
from tastytrade.market_data import get_market_data
from tastytrade.utils import today_in_new_york, is_market_open_on
from tastytrade.instruments import NestedOptionChain
from tastytrade.order import InstrumentType

from config import TIMEOUT_SECONDS


class MarketDataFetcher:
    """Handles all market data fetching operations"""

    def __init__(self):
        self.session = None

    def create_session(self, email, password):
        """Create and return a new Tastytrade session"""
        return Session(email, password)

    def get_spot_price(self, session, symbol):
        """Get current spot price for the given symbol"""
        mdata = get_market_data(session, symbol, InstrumentType.INDEX)
        return float(mdata.last)

    def get_vix_data(self, session):
        """Fetch VIX term structure data"""
        try:
            # Get main VIX
            vix_mdata = get_market_data(session, "VIX", InstrumentType.INDEX)
            vix_current = float(vix_mdata.last)

            # Try to get VIX9D for short-term structure
            try:
                vix9d_mdata = get_market_data(session, "VIX9D", InstrumentType.INDEX)
                vix9d = float(vix9d_mdata.last)
            except:
                vix9d = vix_current * 0.95  # Approximate if not available

            # Try to get VIX3M for long-term structure
            try:
                vix3m_mdata = get_market_data(session, "VIX3M", InstrumentType.INDEX)
                vix3m = float(vix3m_mdata.last)
            except:
                vix3m = vix_current * 1.05  # Approximate if not available

        except Exception as e:
            print(f"⚠️ VIX data fetch failed: {e}, using fallback values")
            # Fallback values if VIX not available
            vix_current = 20.0
            vix9d = 19.0
            vix3m = 21.0

        return {
            'current': vix_current,
            'vix9d': vix9d,
            'vix3m': vix3m,
            'term_structure': vix3m / vix_current,
            'short_term_premium': vix_current / vix9d
        }

    def get_options_chain(self, session, symbol, spot_price, strike_range, dte_limit):
        """Get filtered options chain data"""
        # Find next expiration date
        next_exp = today_in_new_york()
        while not is_market_open_on(next_exp):
            next_exp += timedelta(days=1)

        # Get options chain
        chain = NestedOptionChain.get(session, symbol)
        subs_list = []
        filtered_options = []

        for exp in chain[0].expirations:
            if exp.days_to_expiration <= dte_limit:
                for strike in exp.strikes:
                    price = strike.strike_price
                    if spot_price - strike_range <= price <= spot_price + strike_range:
                        subs_list += [strike.call_streamer_symbol, strike.put_streamer_symbol]
                        filtered_options.append({
                            'strike': price,
                            'call_symbol': strike.call_streamer_symbol,
                            'put_symbol': strike.put_streamer_symbol
                        })

        return subs_list, filtered_options, next_exp.strftime('%Y-%m-%d')

    def fetch_enhanced_market_data(self, email, password, symbol, strike_range, dte_limit):
        """Main function to fetch all required market data"""
        session = Session(email, password)

        # Get spot price
        spot_price = self.get_spot_price(session, symbol)

        # Get VIX data
        vix_data = self.get_vix_data(session)

        # Get options chain
        subs_list, filtered_options, next_exp = self.get_options_chain(
            session, symbol, spot_price, strike_range, dte_limit
        )

        # Get Greeks and Summary data via streaming
        greeks_data, summary_data = self.run_enhanced_streams(session, subs_list)

        return {
            'spot_price': spot_price,
            'vix_data': vix_data,
            'next_exp': next_exp,
            'greeks_data': greeks_data,
            'summary_data': summary_data,
            'filtered_options': filtered_options
        }

    def run_enhanced_streams(self, session, subs_list):
        """Enhanced streaming with better error handling"""
        executor = ThreadPoolExecutor(max_workers=2)
        return executor.submit(self._run_streams_sync, session, subs_list).result()

    def _run_streams_sync(self, session, subs_list):
        """Synchronous wrapper for async streaming"""
        greeks_data, summary_data = {}, {}

        async def collect():
            async with DXLinkStreamer(session) as streamer:
                await streamer.subscribe(Greeks, subs_list)
                await streamer.subscribe(Summary, subs_list)
                await asyncio.wait_for(asyncio.gather(
                    self._collect_greeks(streamer, greeks_data, subs_list),
                    self._collect_summary(streamer, summary_data, subs_list)
                ), timeout=TIMEOUT_SECONDS)

        def runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(collect())
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        t = threading.Thread(target=runner)
        t.start()
        t.join()

        return greeks_data, summary_data

    async def _collect_greeks(self, streamer, greeks_data, subs_list):
        """Collect Greeks data from stream"""
        async for event in streamer.listen(Greeks):
            greeks_data[event.event_symbol] = {
                "symbol": event.event_symbol,
                "delta": float(event.delta or 0),
                "gamma": float(event.gamma or 0),
                "vega": float(event.vega or 0),
                "theta": float(event.theta or 0),
                "iv": float(event.volatility or 0)
            }
            if len(greeks_data) >= len(subs_list):
                break

    async def _collect_summary(self, streamer, summary_data, subs_list):
        """Collect Summary data from stream"""
        async for event in streamer.listen(Summary):
            summary_data[event.event_symbol] = {
                "symbol": event.event_symbol,
                "open_interest": float(event.open_interest or 0)
            }
            if len(summary_data) >= len(subs_list):
                break


# Global instance for easy access
market_data_fetcher = MarketDataFetcher()


def fetch_enhanced_market_data(email, password, symbol, strike_range, dte_limit):
    """Convenience function that wraps the main fetcher method"""
    return market_data_fetcher.fetch_enhanced_market_data(
        email, password, symbol, strike_range, dte_limit
    )