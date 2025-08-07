"""
Chart building functions for SPX Trading Dashboard
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from config import (
    CHART_HEIGHT_MAIN,
    CHART_HEIGHT_SECONDARY,
    CHART_HEIGHT_TARGETS,
    VIX_REGIME_COLORS
)


def build_enhanced_charts(gex_df, spot_price, vix_analysis, momentum_data, flow_history, target_history):
    """Build all chart components"""
    if gex_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            template="plotly_dark",
            height=400
        )
        return {
            'gex': empty_fig,
            'dex': empty_fig,
            'oi': empty_fig,
            'smile': empty_fig,
            'vix_analysis': empty_fig,
            'flow_momentum': empty_fig,
            'targets': empty_fig
        }

    charts = {}

    # 1. GEX Chart
    charts['gex'] = build_gex_chart(gex_df, spot_price)

    # 2. DEX Chart
    charts['dex'] = build_dex_chart(gex_df, spot_price)

    # 3. Open Interest Chart
    charts['oi'] = build_oi_chart(gex_df, spot_price)

    # 4. Volatility Smile Chart
    charts['smile'] = build_volatility_smile_chart(gex_df, spot_price)

    # 5. VIX Analysis Chart
    charts['vix_analysis'] = build_vix_analysis_chart(vix_analysis)

    # 6. Flow Momentum Chart
    charts['flow_momentum'] = build_flow_momentum_chart(momentum_data, flow_history)

    # 7. Dynamic Targets Chart
    charts['targets'] = build_targets_chart(target_history, spot_price)

    return charts


def build_gex_chart(gex_df, spot_price):
    """Build Gamma Exposure (GEX) chart"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=gex_df['strike'],
        y=gex_df['Net_GEX'],
        name='Net GEX',
        marker_color=['red' if x < 0 else 'green' for x in gex_df['Net_GEX']],
        hovertemplate='<b>Strike:</b> %{x}<br><b>Net GEX:</b> %{y:,.0f}<extra></extra>'
    ))

    fig.add_vline(
        x=spot_price,
        line_dash="dash",
        line_color="white",
        annotation_text=f"Spot: {spot_price:.0f}",
        annotation_position="top"
    )

    fig.update_layout(
        title="🎯 Gamma Exposure (GEX)",
        xaxis_title="Strike Price",
        yaxis_title="Net GEX",
        template="plotly_dark",
        height=CHART_HEIGHT_MAIN,
        showlegend=False
    )

    return fig


def build_dex_chart(gex_df, spot_price):
    """Build Delta Exposure (DEX) chart"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=gex_df['strike'],
        y=gex_df['Net_DEX'],
        name='Net DEX',
        marker_color='blue',
        hovertemplate='<b>Strike:</b> %{x}<br><b>Net DEX:</b> %{y:,.0f}<extra></extra>'
    ))

    fig.add_vline(
        x=spot_price,
        line_dash="dash",
        line_color="white",
        annotation_text=f"Spot: {spot_price:.0f}",
        annotation_position="top"
    )

    fig.update_layout(
        title="📊 Delta Exposure (DEX)",
        xaxis_title="Strike Price",
        yaxis_title="Net DEX",
        template="plotly_dark",
        height=CHART_HEIGHT_MAIN,
        showlegend=False
    )

    return fig


def build_oi_chart(gex_df, spot_price):
    """Build Open Interest chart"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=gex_df['strike'],
        y=gex_df['Call_OI'],
        name='Call OI',
        marker_color='green',
        opacity=0.7,
        hovertemplate='<b>Strike:</b> %{x}<br><b>Call OI:</b> %{y:,.0f}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=gex_df['strike'],
        y=-gex_df['Put_OI'],
        name='Put OI',
        marker_color='red',
        opacity=0.7,
        hovertemplate='<b>Strike:</b> %{x}<br><b>Put OI:</b> %{y:,.0f}<extra></extra>'
    ))

    fig.add_vline(
        x=spot_price,
        line_dash="dash",
        line_color="white",
        annotation_text=f"Spot: {spot_price:.0f}",
        annotation_position="top"
    )

    fig.update_layout(
        title="📈 Open Interest",
        xaxis_title="Strike Price",
        yaxis_title="Open Interest",
        template="plotly_dark",
        height=CHART_HEIGHT_MAIN,
        barmode='relative'
    )

    return fig


def build_volatility_smile_chart(gex_df, spot_price):
    """Build volatility smile chart"""
    fig = go.Figure()

    if 'Call_IV' in gex_df.columns and 'Put_IV' in gex_df.columns:
        fig.add_trace(go.Scatter(
            x=gex_df['strike'],
            y=gex_df['Call_IV'],
            mode='lines+markers',
            name='Call IV',
            line_color='green',
            hovertemplate='<b>Strike:</b> %{x}<br><b>Call IV:</b> %{y:.2%}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=gex_df['strike'],
            y=gex_df['Put_IV'],
            mode='lines+markers',
            name='Put IV',
            line_color='red',
            hovertemplate='<b>Strike:</b> %{x}<br><b>Put IV:</b> %{y:.2%}<extra></extra>'
        ))

        fig.add_vline(
            x=spot_price,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Spot: {spot_price:.0f}",
            annotation_position="top"
        )

    fig.update_layout(
        title="😊 Volatility Smile",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility",
        template="plotly_dark",
        height=CHART_HEIGHT_MAIN
    )

    return fig


def build_vix_analysis_chart(vix_analysis):
    """Build VIX analysis gauge chart"""
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=vix_analysis['current_vix'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"VIX ({vix_analysis['regime']})"},
        delta={'reference': 20, 'valueformat': '.1f'},
        gauge={
            'axis': {'range': [None, 50]},
            'bar': {'color': VIX_REGIME_COLORS.get(vix_analysis['regime'], 'gray')},
            'steps': [
                {'range': [0, 15], 'color': "lightgray"},
                {'range': [15, 25], 'color': "gray"},
                {'range': [25, 35], 'color': "orange"},
                {'range': [35, 50], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': vix_analysis['current_vix']
            }
        }
    ))

    # Add text annotations for additional VIX info
    fig.add_annotation(
        x=0.5, y=0.15,
        text=f"Term Structure: {vix_analysis['term_structure']}<br>" +
             f"Ratio: {vix_analysis['term_structure_ratio']:.2f}<br>" +
             f"Momentum: {vix_analysis['vix_momentum']:.2f}",
        showarrow=False,
        font=dict(size=12, color="white"),
        align="center"
    )

    fig.update_layout(
        template="plotly_dark",
        height=CHART_HEIGHT_SECONDARY,
        title=f"🌪️ VIX Regime Analysis - {vix_analysis['term_structure']}"
    )

    return fig


def build_flow_momentum_chart(momentum_data, flow_history):
    """Build flow momentum chart"""
    fig = go.Figure()

    if momentum_data and len(flow_history) > 0:
        df_flow = pd.DataFrame(flow_history)

        fig.add_trace(go.Scatter(
            x=df_flow['timestamp'],
            y=df_flow['total_gex'],
            mode='lines',
            name='Total GEX',
            line_color='green',
            hovertemplate='<b>Time:</b> %{x}<br><b>Total GEX:</b> %{y:,.0f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=df_flow['timestamp'],
            y=df_flow['total_dex'],
            mode='lines',
            name='Total DEX',
            line_color='blue',
            yaxis='y2',
            hovertemplate='<b>Time:</b> %{x}<br><b>Total DEX:</b> %{y:,.0f}<extra></extra>'
        ))

        # Add momentum strength annotation
        if len(df_flow) > 0:
            fig.add_annotation(
                x=df_flow['timestamp'].iloc[-1],
                y=df_flow['total_gex'].iloc[-1],
                text=f"Strength: {momentum_data.get('momentum_strength', 'UNKNOWN')}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="white",
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor="white",
                borderwidth=1
            )

        # Add momentum metrics as text
        momentum_text = (
            f"GEX Short: {momentum_data.get('gex_momentum_short', 0):,.0f}<br>"
            f"GEX Medium: {momentum_data.get('gex_momentum_medium', 0):,.0f}<br>"
            f"DEX Short: {momentum_data.get('dex_momentum_short', 0):,.0f}<br>"
            f"Flow Divergence: {momentum_data.get('flow_divergence', 0):.1f}"
        )

        fig.add_annotation(
            x=0.02, y=0.98,
            text=momentum_text,
            showarrow=False,
            font=dict(size=10, color="white"),
            align="left",
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white",
            borderwidth=1,
            xref="paper",
            yref="paper"
        )

    fig.update_layout(
        title=f"🌊 Flow Momentum ({momentum_data.get('momentum_strength', 'UNKNOWN')})",
        xaxis_title="Time",
        yaxis_title="GEX Flow",
        yaxis2=dict(
            title="DEX Flow",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        template="plotly_dark",
        height=CHART_HEIGHT_SECONDARY,
        hovermode='x unified'
    )

    return fig


def build_targets_chart(target_history, spot_price):
    """Build dynamic targets chart with confidence visualization"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('🎯 Price Targets with Confidence', '⏰ Target Timeline'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )

    if len(target_history) > 0 and target_history[-1]:
        targets = target_history[-1]  # Most recent targets

        # Sort targets by confidence for better visualization
        sorted_targets = sorted(targets, key=lambda x: x['confidence'], reverse=True)

        # Plot targets as horizontal lines with confidence-based styling
        for i, target in enumerate(sorted_targets):
            opacity = max(0.3, target['confidence'] / 100)
            line_width = max(1, int(target['confidence'] / 20))
            color = 'green' if target['direction'] == 'UPSIDE' else 'red'

            # Main target line
            fig.add_hline(
                y=target['strike'],
                line_dash="dash",
                line_color=color,
                opacity=opacity,
                line_width=line_width,
                annotation_text=f"{target['strike']:.0f} ({target['confidence']:.0f}%) - {target['gex_type']}",
                annotation_position="right",
                row=1, col=1
            )

            # Add confidence bars in the bottom subplot
            fig.add_trace(go.Bar(
                x=[f"{target['strike']:.0f}"],
                y=[target['confidence']],
                name=f"{target['strike']:.0f}",
                marker_color=color,
                opacity=opacity,
                text=[f"{target['confidence']:.0f}%"],
                textposition='auto',
                showlegend=False,
                hovertemplate=f'<b>Strike:</b> {target["strike"]:.0f}<br>' +
                              f'<b>Confidence:</b> {target["confidence"]:.1f}%<br>' +
                              f'<b>Direction:</b> {target["direction"]}<br>' +
                              f'<b>Type:</b> {target["gex_type"]}<br>' +
                              f'<b>Expected Time:</b> {target["expected_time_hours"]:.1f}h<extra></extra>'
            ), row=2, col=1)

        # Add trend lines connecting targets over time if we have history
        if len(target_history) > 1:
            # Track the strongest target over time
            timestamps = list(range(len(target_history)))
            strongest_strikes = []

            for targets_set in target_history:
                if targets_set:
                    strongest = max(targets_set, key=lambda x: x['confidence'])
                    strongest_strikes.append(strongest['strike'])
                else:
                    strongest_strikes.append(None)

            # Remove None values for plotting
            valid_data = [(t, s) for t, s in zip(timestamps, strongest_strikes) if s is not None]
            if len(valid_data) > 1:
                times, strikes = zip(*valid_data)

                fig.add_trace(go.Scatter(
                    x=times,
                    y=strikes,
                    mode='lines+markers',
                    name='Target Evolution',
                    line_color='yellow',
                    opacity=0.7,
                    showlegend=False
                ), row=1, col=1)

    # Current price line (most prominent)
    fig.add_hline(
        y=spot_price,
        line_color="white",
        line_width=4,
        annotation_text=f"Current: {spot_price:.0f}",
        annotation_position="left",
        row=1, col=1
    )

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=CHART_HEIGHT_TARGETS,
        showlegend=False
    )

    # Update axes
    fig.update_xaxes(title_text="Target Rank", row=2, col=1)
    fig.update_yaxes(title_text="Price Level", row=1, col=1)
    fig.update_yaxes(title_text="Confidence %", row=2, col=1, range=[0, 100])

    return fig


def create_empty_chart(title="No data available"):
    """Create an empty chart placeholder"""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=400,
        annotations=[
            dict(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=20, color="gray")
            )
        ]
    )
    return fig