"""
Trading signal generation and display components for SPX Trading Dashboard
"""

from dash import html
import dash_bootstrap_components as dbc
from config import MOMENTUM_THRESHOLDS


def generate_trading_signals(targets, momentum_data, vix_analysis, confidence_threshold):
    """Generate actionable trading signals"""
    signals = []

    # Filter high-confidence targets
    high_conf_targets = [t for t in targets if t['confidence'] >= confidence_threshold]

    if not high_conf_targets:
        return [{
            'signal_type': 'WAIT',
            'message': f'No high-confidence targets above {confidence_threshold}% threshold',
            'confidence': 0,
            'action': 'WAIT'
        }]

    # Primary directional signal based on strongest target + momentum
    primary_target = high_conf_targets[0]
    momentum_strength = momentum_data['momentum_strength']

    # Determine primary signal
    if primary_target['direction'] == 'UPSIDE':
        if momentum_data['dex_momentum_short'] > 0 and vix_analysis['vix_momentum'] < 0:
            signal_type = 'STRONG_BULLISH'
            action = 'BUY_CALLS'
        elif momentum_data['dex_momentum_short'] > 0:
            signal_type = 'BULLISH'
            action = 'BUY_CALLS'
        else:
            signal_type = 'WEAK_BULLISH'
            action = 'SELL_PUTS'
    else:  # DOWNSIDE
        if momentum_data['dex_momentum_short'] < 0 and vix_analysis['vix_momentum'] > 0:
            signal_type = 'STRONG_BEARISH'
            action = 'BUY_PUTS'
        elif momentum_data['dex_momentum_short'] < 0:
            signal_type = 'BEARISH'
            action = 'BUY_PUTS'
        else:
            signal_type = 'WEAK_BEARISH'
            action = 'SELL_CALLS'

    # Add primary signal
    signals.append({
        'signal_type': signal_type,
        'target_strike': primary_target['strike'],
        'confidence': primary_target['confidence'],
        'action': action,
        'gex_type': primary_target['gex_type'],
        'expected_time': f"{primary_target['expected_time_hours']:.1f}h",
        'message': f"{signal_type}: Target {primary_target['strike']:.0f} ({primary_target['confidence']:.0f}% conf)"
    })

    # Secondary signals for counter-moves
    opposing_targets = [t for t in high_conf_targets if t['direction'] != primary_target['direction']]
    if opposing_targets:
        secondary_target = opposing_targets[0]
        signals.append({
            'signal_type': 'HEDGE_OPPORTUNITY',
            'target_strike': secondary_target['strike'],
            'confidence': secondary_target['confidence'],
            'action': 'HEDGE_PRIMARY',
            'gex_type': secondary_target['gex_type'],
            'expected_time': f"{secondary_target['expected_time_hours']:.1f}h",
            'message': f"Hedge at {secondary_target['strike']:.0f} ({secondary_target['confidence']:.0f}% conf)"
        })

    # VIX regime warnings
    if vix_analysis['regime'] == 'PANIC_VOL':
        signals.append({
            'signal_type': 'VIX_WARNING',
            'message': f"⚠️ PANIC VOL REGIME (VIX {vix_analysis['current_vix']:.1f}) - Reduce position sizes",
            'confidence': 90,
            'action': 'REDUCE_SIZE'
        })
    elif vix_analysis['term_structure_ratio'] < 0.9:
        signals.append({
            'signal_type': 'BACKWARDATION_WARNING',
            'message': f"⚠️ VIX in backwardation - Potential volatility spike ahead",
            'confidence': 75,
            'action': 'BUY_PROTECTION'
        })

    # Flow divergence signals
    flow_divergence = momentum_data.get('flow_divergence', 0)
    if abs(flow_divergence) >= 1.5:
        divergence_type = 'BEARISH' if flow_divergence > 0 else 'BULLISH'
        signals.append({
            'signal_type': f'FLOW_DIVERGENCE_{divergence_type}',
            'message': f"📊 Flow-Price Divergence: {divergence_type.title()} signal detected",
            'confidence': 65,
            'action': f'WATCH_FOR_{divergence_type}_REVERSAL'
        })

    # Momentum strength warnings
    if momentum_data['momentum_strength'] == 'WEAK':
        signals.append({
            'signal_type': 'LOW_MOMENTUM_WARNING',
            'message': "⚡ Low momentum environment - Consider range-bound strategies",
            'confidence': 60,
            'action': 'USE_RANGE_STRATEGIES'
        })

    return signals


def build_signal_display(signals, targets):
    """Build the signal display component"""
    if not signals:
        return html.Div("No signals generated", className="text-muted")

    signal_cards = []

    for signal in signals:
        # Color coding and icons
        signal_type = signal.get('signal_type', '')

        if 'BULLISH' in signal_type:
            if 'STRONG' in signal_type:
                color, icon = 'success', '🚀'
            elif 'WEAK' in signal_type:
                color, icon = 'success', '📈'
            else:
                color, icon = 'success', '📊'

        elif 'BEARISH' in signal_type:
            if 'STRONG' in signal_type:
                color, icon = 'danger', '💥'
            elif 'WEAK' in signal_type:
                color, icon = 'danger', '📉'
            else:
                color, icon = 'danger', '📊'

        elif 'WARNING' in signal_type:
            color, icon = 'warning', '⚠️'

        elif 'HEDGE' in signal_type:
            color, icon = 'info', '🛡️'

        elif 'DIVERGENCE' in signal_type:
            color, icon = 'primary', '🔄'

        elif 'WAIT' in signal_type:
            color, icon = 'secondary', '⏸️'

        else:
            color, icon = 'secondary', '📋'

        # Build card content
        card_content = [
            html.H6(f"{icon} {signal_type.replace('_', ' ')}", className=f"text-{color}"),
            html.P(signal.get('message', ''), className="mb-1"),
        ]

        # Add target strike if available
        if signal.get('target_strike'):
            card_content.append(
                html.P(f"🎯 Target: {signal['target_strike']:.0f}", className="mb-1 fw-bold")
            )

        # Add confidence if available
        if signal.get('confidence', 0) > 0:
            card_content.append(
                html.P(f"📊 Confidence: {signal['confidence']:.0f}%", className="mb-1")
            )

        # Add expected time if available
        if signal.get('expected_time'):
            card_content.append(
                html.P(f"⏰ Time: {signal['expected_time']}", className="mb-1")
            )

        # Add action
        card_content.append(
            html.Small(f"💡 Action: {signal.get('action', 'N/A')}", className="text-muted")
        )

        signal_cards.append(
            dbc.Card(
                dbc.CardBody(card_content),
                color=color,
                outline=True,
                className="mb-2"
            )
        )

    # Add summary card if multiple signals
    if len(signals) > 1:
        primary_signals = [s for s in signals if
                           s.get('signal_type') not in ['VIX_WARNING', 'BACKWARDATION_WARNING', 'LOW_MOMENTUM_WARNING']]

        if primary_signals:
            strongest_signal = max(primary_signals, key=lambda x: x.get('confidence', 0))

            summary_card = dbc.Card(
                dbc.CardBody([
                    html.H6("📋 Summary", className="text-info"),
                    html.P(f"Primary: {strongest_signal.get('signal_type', 'N/A').replace('_', ' ')}",
                           className="mb-1"),
                    html.P(f"Confidence: {strongest_signal.get('confidence', 0):.0f}%", className="mb-1"),
                    html.Small(f"Total Signals: {len(signals)}", className="text-muted")
                ]),
                color="info",
                outline=True,
                className="mb-3"
            )
            signal_cards.insert(0, summary_card)

    return html.Div(signal_cards)


def classify_signal_strength(confidence, momentum_strength, vix_regime):
    """Classify overall signal strength"""
    base_score = confidence / 100

    # Momentum adjustment
    if momentum_strength == 'STRONG':
        momentum_multiplier = 1.2
    elif momentum_strength == 'MODERATE':
        momentum_multiplier = 1.0
    else:
        momentum_multiplier = 0.8

    # VIX regime adjustment
    vix_multipliers = {
        'LOW_VOL': 1.1,
        'NORMAL_VOL': 1.0,
        'HIGH_VOL': 0.9,
        'PANIC_VOL': 0.7
    }
    vix_multiplier = vix_multipliers.get(vix_regime, 1.0)

    final_score = base_score * momentum_multiplier * vix_multiplier

    if final_score >= 0.8:
        return 'VERY_STRONG'
    elif final_score >= 0.65:
        return 'STRONG'
    elif final_score >= 0.5:
        return 'MODERATE'
    elif final_score >= 0.35:
        return 'WEAK'
    else:
        return 'VERY_WEAK'


def generate_risk_warnings(signals, vix_analysis, momentum_data):
    """Generate risk management warnings"""
    warnings = []

    # High VIX warnings
    if vix_analysis['current_vix'] > 30:
        warnings.append({
            'type': 'HIGH_VOLATILITY',
            'message': f"High volatility environment (VIX {vix_analysis['current_vix']:.1f}) - Use smaller position sizes",
            'severity': 'HIGH'
        })

    # Backwardation warning
    if vix_analysis['term_structure_ratio'] < 0.95:
        warnings.append({
            'type': 'TERM_STRUCTURE',
            'message': "VIX term structure in backwardation - Potential volatility expansion",
            'severity': 'MEDIUM'
        })

    # Low momentum warning
    if momentum_data['momentum_strength'] == 'WEAK':
        warnings.append({
            'type': 'LOW_MOMENTUM',
            'message': "Low momentum environment - Avoid directional bets",
            'severity': 'MEDIUM'
        })

    # Conflicting signals warning
    bullish_signals = len([s for s in signals if 'BULLISH' in s.get('signal_type', '')])
    bearish_signals = len([s for s in signals if 'BEARISH' in s.get('signal_type', '')])

    if bullish_signals > 0 and bearish_signals > 0:
        warnings.append({
            'type': 'CONFLICTING_SIGNALS',
            'message': f"Mixed signals detected ({bullish_signals} bullish, {bearish_signals} bearish)",
            'severity': 'LOW'
        })

    return warnings


def format_action_recommendation(action):
    """Format action recommendations for display"""
    action_mappings = {
        'BUY_CALLS': '📈 Buy Calls',
        'BUY_PUTS': '📉 Buy Puts',
        'SELL_CALLS': '📤 Sell Calls',
        'SELL_PUTS': '📥 Sell Puts',
        'HEDGE_PRIMARY': '🛡️ Hedge Position',
        'REDUCE_SIZE': '⚖️ Reduce Size',
        'BUY_PROTECTION': '🛡️ Buy Protection',
        'WAIT': '⏸️ Wait',
        'WATCH_FOR_BULLISH_REVERSAL': '👀 Watch Bullish',
        'WATCH_FOR_BEARISH_REVERSAL': '👀 Watch Bearish',
        'USE_RANGE_STRATEGIES': '📊 Range Strategies'
    }

    return action_mappings.get(action, f"📋 {action.replace('_', ' ').title()}")