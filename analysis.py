"""
Analysis functions for VIX regime analysis, momentum calculations, and dynamic target generation
"""

import pandas as pd
import numpy as np
from datetime import datetime
from config import (
    VIX_THRESHOLDS, VIX_REGIME_COLORS, MAX_FLOW_HISTORY,
    MAX_VIX_HISTORY, MAX_TARGET_HISTORY, MOMENTUM_THRESHOLDS
)

# Global data storage for momentum calculations
flow_history = []
vix_history = []
target_history = []


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

    # Keep only last N entries
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


def analyze_vix_regime(vix_data, spot_price):
    """Analyze VIX regime and implications"""
    global vix_history

    current_vix = vix_data['current']

    # VIX regime classification
    if current_vix < VIX_THRESHOLDS['LOW_VOL']:
        regime = 'LOW_VOL'
        regime_color = VIX_REGIME_COLORS['LOW_VOL']
    elif current_vix < VIX_THRESHOLDS['NORMAL_VOL']:
        regime = 'NORMAL_VOL'
        regime_color = VIX_REGIME_COLORS['NORMAL_VOL']
    elif current_vix < VIX_THRESHOLDS['HIGH_VOL']:
        regime = 'HIGH_VOL'
        regime_color = VIX_REGIME_COLORS['HIGH_VOL']
    else:
        regime = 'PANIC_VOL'
        regime_color = VIX_REGIME_COLORS['PANIC_VOL']

    # Term structure analysis
    if vix_data['term_structure'] > 1.1:
        term_structure = 'STEEP_CONTANGO'
    elif vix_data['term_structure'] > 1.05:
        term_structure = 'CONTANGO'
    elif vix_data['term_structure'] < 0.95:
        term_structure = 'BACKWARDATION'
    else:
        term_structure = 'FLAT'

    # VIX momentum if we have history
    vix_momentum = 0
    if len(vix_history) >= 6:
        df_vix = pd.DataFrame(vix_history)
        recent_vix = df_vix.tail(6)
        vix_momentum = (recent_vix['vix'].iloc[-1] - recent_vix['vix'].iloc[0]) / 6

    return {
        'regime': regime,
        'regime_color': regime_color,
        'current_vix': current_vix,
        'term_structure': term_structure,
        'term_structure_ratio': vix_data['term_structure'],
        'vix_momentum': vix_momentum,
        'short_term_premium': vix_data['short_term_premium']
    }


def calculate_dynamic_targets(gex_df, spot_price, vix_analysis, momentum_data, hedge_radius, decay_factor, weights):
    """Calculate dynamic price targets with confidence scores"""
    global target_history

    if gex_df.empty:
        return []

    w1, w2, w3, w4, w5 = weights
    current_vix = vix_analysis['current_vix']

    # Key support/resistance levels from GEX
    gex_df['abs_gex'] = abs(gex_df['Net_GEX'])
    major_levels = gex_df.nlargest(5, 'abs_gex')[['strike', 'Net_GEX', 'Total_OI']].copy()

    targets = []

    for _, level in major_levels.iterrows():
        strike = level['strike']
        distance = abs(strike - spot_price)

        # Skip if too close to current price
        if distance < 5:
            continue

        # Base confidence from GEX strength and OI
        gex_strength = abs(level['Net_GEX']) / gex_df['abs_gex'].max()
        oi_strength = level['Total_OI'] / gex_df['Total_OI'].max()
        base_confidence = (gex_strength * 0.6 + oi_strength * 0.4) * 100

        # Adjust confidence based on VIX regime
        vix_adjustment = 1.0
        if vix_analysis['regime'] == 'LOW_VOL':
            vix_adjustment = 1.2  # Higher confidence in low vol
        elif vix_analysis['regime'] == 'HIGH_VOL':
            vix_adjustment = 0.8  # Lower confidence in high vol
        elif vix_analysis['regime'] == 'PANIC_VOL':
            vix_adjustment = 0.6  # Much lower confidence in panic

        # Adjust for momentum alignment
        momentum_alignment = 1.0
        if strike > spot_price:  # Upside target
            if momentum_data['dex_momentum_short'] > 0:
                momentum_alignment = 1.3
            elif momentum_data['dex_momentum_short'] < -500:
                momentum_alignment = 0.7
        else:  # Downside target
            if momentum_data['dex_momentum_short'] < 0:
                momentum_alignment = 1.3
            elif momentum_data['dex_momentum_short'] > 500:
                momentum_alignment = 0.7

        # Distance decay (closer targets more likely)
        distance_factor = np.exp(-distance / (current_vix * 2))  # VIX-adjusted distance

        # Final confidence calculation
        final_confidence = min(95, base_confidence * vix_adjustment * momentum_alignment * distance_factor)

        # Time to target estimation (based on VIX and distance)
        expected_time_hours = distance / (current_vix * 0.5)  # Rough approximation

        # Target classification
        if strike > spot_price:
            direction = 'UPSIDE'
            gex_type = 'RESISTANCE' if level['Net_GEX'] > 0 else 'MAGNET'
        else:
            direction = 'DOWNSIDE'
            gex_type = 'SUPPORT' if level['Net_GEX'] < 0 else 'VOID'

        targets.append({
            'strike': strike,
            'direction': direction,
            'distance': distance,
            'confidence': final_confidence,
            'gex_type': gex_type,
            'gex_value': level['Net_GEX'],
            'oi_value': level['Total_OI'],
            'expected_time_hours': expected_time_hours
        })

    # Sort by confidence descending
    sorted_targets = sorted(targets, key=lambda x: x['confidence'], reverse=True)

    # Store targets for PineScript and historical analysis
    if sorted_targets:
        target_history.append(sorted_targets)
        target_history = target_history[-MAX_TARGET_HISTORY:]
        print(f"✅ Updated target_history with {len(sorted_targets)} targets")

    return sorted_targets


def get_flow_history():
    """Get current flow history"""
    return flow_history


def get_vix_history():
    """Get current VIX history"""
    return vix_history


def get_target_history():
    """Get current target history"""
    return target_history


def clear_historical_data():
    """Clear all historical data - useful for testing or reset"""
    global flow_history, vix_history, target_history
    flow_history = []
    vix_history = []
    target_history = []


def get_flow_momentum_strength(momentum_data):
    """Get human-readable momentum strength description"""
    strength = momentum_data.get('momentum_strength', 'UNKNOWN')

    descriptions = {
        'STRONG': '🔥 Strong momentum - High conviction moves expected',
        'MODERATE': '⚡ Moderate momentum - Directional bias present',
        'WEAK': '🔄 Weak momentum - Consolidation or reversal possible',
        'BUILDING': '🌱 Building momentum - Early stage development'
    }

    return descriptions.get(strength, f'❓ {strength} momentum')


def calculate_momentum_divergence_signal(momentum_data):
    """Calculate divergence signal strength"""
    divergence = momentum_data.get('flow_divergence', 0)

    if abs(divergence) >= 2:
        return 'STRONG_DIVERGENCE'
    elif abs(divergence) >= 1:
        return 'MODERATE_DIVERGENCE'
    else:
        return 'NO_DIVERGENCE'


def get_vix_regime_implications(vix_analysis):
    """Get trading implications for current VIX regime"""
    regime = vix_analysis['regime']

    implications = {
        'LOW_VOL': {
            'description': 'Low volatility environment - premium selling favored',
            'strategies': ['Sell options premium', 'Iron condors', 'Covered calls'],
            'risks': ['Vol expansion risk', 'Complacency']
        },
        'NORMAL_VOL': {
            'description': 'Normal volatility - balanced approach',
            'strategies': ['Directional trades', 'Straddles/strangles', 'Delta hedging'],
            'risks': ['Regime changes', 'Time decay']
        },
        'HIGH_VOL': {
            'description': 'High volatility - directional opportunities',
            'strategies': ['Buy options', 'Protective puts', 'Volatility plays'],
            'risks': ['Vol crush', 'Whipsaws']
        },
        'PANIC_VOL': {
            'description': 'Panic volatility - extreme caution required',
            'strategies': ['Cash preservation', 'Hedging', 'Small positions'],
            'risks': ['Liquidity issues', 'Gap moves', 'Margin calls']
        }
    }

    return implications.get(regime, {
        'description': 'Unknown regime',
        'strategies': ['Wait for clarity'],
        'risks': ['Uncertainty']
    })