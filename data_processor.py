"""
Options data processing and calculations for SPX Trading Dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime
from config import MAX_FLOW_HISTORY, MAX_VIX_HISTORY, MAX_TARGET_HISTORY

# Global data storage for momentum calculations
flow_history = []
vix_history = []
target_history = []


def process_options_data(greeks_data, summary_data, filtered_options, dte_limit):
    """Process raw options data into GEX DataFrame"""
    if not greeks_data or not summary_data:
        return pd.DataFrame()

    df_greeks = pd.DataFrame(greeks_data.values())
    df_summary = pd.DataFrame(summary_data.values())

    # Convert all numeric columns to float immediately
    numeric_cols = ['delta', 'gamma', 'vega', 'theta', 'iv', 'open_interest']
    for col in numeric_cols:
        if col in df_greeks.columns:
            df_greeks[col] = pd.to_numeric(df_greeks[col], errors='coerce').fillna(0).astype(float)
        if col in df_summary.columns:
            df_summary[col] = pd.to_numeric(df_summary[col], errors='coerce').fillna(0).astype(float)

    df = pd.merge(df_greeks, df_summary, on='symbol', how='inner')
    df_options = pd.DataFrame(filtered_options)

    # Convert strike prices to float too
    df_options['strike'] = pd.to_numeric(df_options['strike'], errors='coerce').astype(float)

    # Process calls
    calls = df.merge(df_options, left_on='symbol', right_on='call_symbol', how='inner')
    calls['Call_GEX'] = calls['gamma'].astype(float) * calls['open_interest'].astype(float) * 100
    calls['Call_DEX'] = calls['delta'].astype(float) * calls['open_interest'].astype(float) * 100
    calls['Call_IV'] = calls['iv'].astype(float)
    calls_grouped = calls.groupby('strike').agg({
        'Call_GEX': 'sum',
        'Call_DEX': 'sum',
        'open_interest': 'sum',
        'Call_IV': 'mean'
    }).rename(columns={'open_interest': 'Call_OI'}).reset_index()

    # Process puts
    puts = df.merge(df_options, left_on='symbol', right_on='put_symbol', how='inner')
    puts['Put_GEX'] = puts['gamma'].astype(float) * puts['open_interest'].astype(float) * 100
    puts['Put_DEX'] = puts['delta'].astype(float) * puts['open_interest'].astype(float) * 100
    puts['Put_IV'] = puts['iv'].astype(float)
    puts_grouped = puts.groupby('strike').agg({
        'Put_GEX': 'sum',
        'Put_DEX': 'sum',
        'open_interest': 'sum',
        'Put_IV': 'mean'
    }).rename(columns={'open_interest': 'Put_OI'}).reset_index()

    # Merge and calculate final metrics
    gex_df = pd.merge(calls_grouped, puts_grouped, on='strike', how='outer').fillna(0)
    gex_df['Net_GEX'] = gex_df['Call_GEX'].astype(float) - gex_df['Put_GEX'].astype(float)
    gex_df['Net_DEX'] = gex_df['Call_DEX'].astype(float) + gex_df['Put_DEX'].astype(float)
    gex_df['Charm'] = gex_df['Net_DEX'].diff().fillna(0).astype(float) / max(1, dte_limit)
    gex_df['Cumulative_GEX'] = gex_df['Net_GEX'].astype(float).cumsum()
    gex_df['Total_OI'] = gex_df['Call_OI'].astype(float) + gex_df['Put_OI'].astype(float)

    return gex_df.sort_values('strike').reset_index(drop=True)


def calculate_flip_strike(gex_df, spot_price):
    """Calculate the gamma flip strike"""
    if gex_df.empty:
        return spot_price

    flip_idx = gex_df['Cumulative_GEX'].abs().idxmin()
    return gex_df.loc[flip_idx, 'strike']


def store_historical_data(gex_df, vix_data, spot_price, timestamp):
    """Store historical data for momentum calculations"""
    global flow_history, vix_history

    if gex_df.empty:
        return

    # Calculate aggregate flow metrics
    total_gex = gex_df['Net_GEX'].sum()
    total_dex = gex_df['Net_DEX'].sum()
    total_charm = gex_df['Charm'].sum()

    flow_entry = {
        'timestamp': timestamp,
        'spot_price': spot_price,
        'total_gex': total_gex,
        'total_dex': total_dex,
        'total_charm': total_charm,
        'gex_near_money': gex_df[(abs(gex_df['strike'] - spot_price) <= 25)]['Net_GEX'].sum(),
        'dex_near_money': gex_df[(abs(gex_df['strike'] - spot_price) <= 25)]['Net_DEX'].sum()
    }

    vix_entry = {
        'timestamp': timestamp,
        'vix': vix_data['current'],
        'vix9d': vix_data['vix9d'],
        'vix3m': vix_data['vix3m'],
        'term_structure': vix_data['term_structure'],
        'short_term_premium': vix_data['short_term_premium']
    }

    flow_history.append(flow_entry)
    vix_history.append(vix_entry)

    # Keep only recent entries to prevent memory issues
    flow_history = flow_history[-MAX_FLOW_HISTORY:]
    vix_history = vix_history[-MAX_VIX_HISTORY:]


def calculate_flow_momentum(gex_df, spot_price):
    """Calculate momentum indicators for flow data"""
    global flow_history

    if len(flow_history) < 6:  # Need at least 6 data points
        return {
            'gex_momentum_short': 0,
            'gex_momentum_medium': 0,
            'dex_momentum_short': 0,
            'dex_momentum_medium': 0,
            'flow_divergence': 0,
            'momentum_strength': 'BUILDING'
        }

    df_history = pd.DataFrame(flow_history)

    # Calculate momentum over different periods
    periods = [3, 6, 12]  # Short, medium, long lookbacks
    momentum_data = {}

    for period in periods:
        if len(df_history) >= period:
            recent = df_history.tail(period)

            # GEX momentum (rate of change)
            gex_change = (recent['total_gex'].iloc[-1] - recent['total_gex'].iloc[0]) / period
            dex_change = (recent['total_dex'].iloc[-1] - recent['total_dex'].iloc[0]) / period

            momentum_data[f'gex_momentum_{period}'] = gex_change
            momentum_data[f'dex_momentum_{period}'] = dex_change

    # Flow vs Price divergence
    if len(df_history) >= 12:
        recent_12 = df_history.tail(12)
        price_change = recent_12['spot_price'].iloc[-1] - recent_12['spot_price'].iloc[0]
        flow_change = recent_12['dex_near_money'].iloc[-1] - recent_12['dex_near_money'].iloc[0]

        # Normalize changes
        price_direction = 1 if price_change > 0 else -1
        flow_direction = 1 if flow_change > 0 else -1

        momentum_data['flow_divergence'] = price_direction - flow_direction  # -2 to +2

    # Overall momentum strength classification
    from config import MOMENTUM_THRESHOLDS
    short_strength = abs(momentum_data.get('gex_momentum_3', 0)) + abs(momentum_data.get('dex_momentum_3', 0))

    if short_strength > MOMENTUM_THRESHOLDS['STRONG']:
        strength = 'STRONG'
    elif short_strength > MOMENTUM_THRESHOLDS['MODERATE']:
        strength = 'MODERATE'
    else:
        strength = 'WEAK'

    momentum_data['momentum_strength'] = strength
    momentum_data['gex_momentum_short'] = momentum_data.get('gex_momentum_3', 0)
    momentum_data['gex_momentum_medium'] = momentum_data.get('gex_momentum_6', 0)
    momentum_data['dex_momentum_short'] = momentum_data.get('dex_momentum_3', 0)
    momentum_data['dex_momentum_medium'] = momentum_data.get('dex_momentum_6', 0)

    return momentum_data


def store_targets(targets):
    """Store targets for PineScript and historical analysis"""
    global target_history

    if targets:
        target_history.append(targets)
        target_history = target_history[-MAX_TARGET_HISTORY:]
        print(f"✅ Updated target_history with {len(targets)} targets")


def get_historical_data():
    """Get stored historical data"""
    return {
        'flow_history': flow_history,
        'vix_history': vix_history,
        'target_history': target_history
    }


def clear_historical_data():
    """Clear all historical data (useful for testing/reset)"""
    global flow_history, vix_history, target_history
    flow_history.clear()
    vix_history.clear()
    target_history.clear()


def calculate_key_levels(gex_df, num_levels=8):
    """Calculate key support/resistance levels from GEX data"""
    if gex_df.empty:
        return pd.DataFrame()

    gex_df['abs_gex'] = abs(gex_df['Net_GEX'])
    return gex_df.nlargest(num_levels, 'abs_gex')[['strike', 'Net_GEX', 'Total_OI']].copy()


def calculate_near_money_metrics(gex_df, spot_price, radius=25):
    """Calculate GEX/DEX metrics near current price"""
    if gex_df.empty:
        return {
            'near_money_gex': 0,
            'near_money_dex': 0,
            'near_money_oi': 0
        }

    near_money = gex_df[abs(gex_df['strike'] - spot_price) <= radius]

    return {
        'near_money_gex': near_money['Net_GEX'].sum(),
        'near_money_dex': near_money['Net_DEX'].sum(),
        'near_money_oi': near_money['Total_OI'].sum()
    }


def validate_gex_dataframe(gex_df):
    """Validate and clean GEX DataFrame"""
    if gex_df.empty:
        return gex_df

    # Ensure all required columns exist
    required_columns = [
        'strike', 'Net_GEX', 'Net_DEX', 'Charm', 'Cumulative_GEX',
        'Total_OI', 'Call_GEX', 'Put_GEX', 'Call_OI', 'Put_OI'
    ]

    for col in required_columns:
        if col not in gex_df.columns:
            gex_df[col] = 0.0

    # Ensure numeric types
    numeric_columns = [col for col in required_columns if col != 'strike']
    for col in numeric_columns:
        gex_df[col] = pd.to_numeric(gex_df[col], errors='coerce').fillna(0).astype(float)

    # Sort by strike
    gex_df = gex_df.sort_values('strike').reset_index(drop=True)

    return gex_df