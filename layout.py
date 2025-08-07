"""
Dash layout components for SPX Trading Dashboard
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
from config import (
    DEFAULT_SYMBOL, DEFAULT_STRIKE_RANGE, DEFAULT_DTE_LIMIT,
    DEFAULT_HEDGE_RADIUS, DEFAULT_DECAY_FACTOR, DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_WEIGHTS, REFRESH_INTERVAL
)


def create_login_section():
    """Create the login form section"""
    return dbc.Col([
        html.H4("🔑 Login to Tastytrade"),
        html.Label("Username:"),
        dbc.Input(id='email', placeholder="Email", type="email", className="mb-2"),
        html.Label("Password:"),
        dbc.Input(id='password', placeholder="Password", type="password", className="mb-2"),
        dbc.Button("Login", id='login-btn', color="primary", className="mb-3 w-100"),
        dbc.Alert(id='login-alert', is_open=False, duration=4000),
    ], width=3)


def create_settings_section():
    """Create the trading settings section"""
    return html.Div([
        html.Hr(),
        html.H5("⚙️ Settings"),

        # Basic Settings
        html.Label("Symbol (SPX, SPY, ESU5):"),
        dbc.Input(id='symbol', placeholder="Symbol (e.g., SPX)",
                  value=DEFAULT_SYMBOL, className="mb-2"),

        html.Label("Strike Range (+/-):"),
        dbc.Input(id='strike_range', placeholder="Strike Range", type="number",
                  value=DEFAULT_STRIKE_RANGE, className="mb-2"),

        html.Label("DTE Limit (days):"),
        dbc.Input(id='dte_limit', placeholder="DTE Limit", type="number",
                  value=DEFAULT_DTE_LIMIT, className="mb-2"),

        html.Label("Hedging Radius (pts):"),
        dbc.Input(id='hedge_radius', placeholder="Radius (e.g., 50)", type="number",
                  value=DEFAULT_HEDGE_RADIUS, className="mb-2"),

        html.Label("Decay Factor (0.01-0.1):"),
        dbc.Input(id='decay_factor', placeholder="Decay (e.g., 0.05)", type="number",
                  value=DEFAULT_DECAY_FACTOR, step=0.01, className="mb-2"),

        html.Label("Target Confidence Threshold:"),
        dbc.Input(id='confidence_threshold', placeholder="Confidence %", type="number",
                  value=DEFAULT_CONFIDENCE_THRESHOLD, className="mb-3"),
    ])


def create_weight_controls():
    """Create the weight control section"""
    return html.Div([
        # Weight controls
        html.Label("Gamma Weight (w1):"),
        dbc.Input(id='w1', placeholder="Gamma weight", type="number",
                  value=DEFAULT_WEIGHTS['w1'], step=0.1, className="mb-2"),

        html.Label("Delta Weight (w2):"),
        dbc.Input(id='w2', placeholder="Delta weight", type="number",
                  value=DEFAULT_WEIGHTS['w2'], step=0.1, className="mb-2"),

        html.Label("Charm Weight (w3):"),
        dbc.Input(id='w3', placeholder="Charm weight", type="number",
                  value=DEFAULT_WEIGHTS['w3'], step=0.1, className="mb-2"),

        html.Label("Vega Weight (w4):"),
        dbc.Input(id='w4', placeholder="Vega weight", type="number",
                  value=DEFAULT_WEIGHTS['w4'], step=0.1, className="mb-2"),

        html.Label("VIX Weight (w5):"),
        dbc.Input(id='w5', placeholder="VIX weight", type="number",
                  value=DEFAULT_WEIGHTS['w5'], step=0.1, className="mb-3"),

        dbc.Button("🔄 Update Settings", id='update-settings-btn',
                   color="secondary", className="w-100 mb-3"),
    ])


def create_signal_section():
    """Create the live signals section"""
    return html.Div([
        html.Hr(),
        html.H5("🎯 Live Signals"),
        html.Div(id='signal-display', className="mb-3"),
    ])


def create_pinescript_section():
    """Create the PineScript generation section"""
    return html.Div([
        dbc.Button("📋 Generate PineScript", id="copy-pine-btn",
                   color="secondary", className="w-100 mb-2"),
        dcc.Textarea(id="pine-textarea",
                     style={"width": "100%", "height": 200},
                     readOnly=True, value=""),
        dcc.Clipboard(id="clipboard", target_id="pine-textarea",
                      title="Copy Pinescript",
                      style={
                          "display": "inline-block",
                          "fontSize": 24,
                          "cursor": "pointer",
                          "marginTop": "5px"
                      }),
    ])


def create_sidebar():
    """Create the complete sidebar with all controls"""
    return dbc.Col([
        create_login_section(),
        create_settings_section(),
        create_weight_controls(),
        create_signal_section(),
        create_pinescript_section(),
    ], width=3)


def create_main_content():
    """Create the main dashboard content area"""
    return dbc.Col([
        dcc.Loading(
            id="loading-dashboard",
            type="circle",
            children=html.Div(id='dashboard-content')
        ),
        dcc.Interval(id='interval-component',
                     interval=REFRESH_INTERVAL,
                     n_intervals=0)
    ], width=9)


def create_app_layout():
    """Create the complete application layout"""
    return dbc.Container([
        html.H1("🎯 SPX Ultimate Trading Dashboard - VIX + Flow + Targets",
                className="text-center my-4"),

        # Session data storage
        dcc.Store(id='session-data', storage_type='session'),

        # Main layout row
        dbc.Row([
            create_sidebar(),
            create_main_content()
        ])
    ], fluid=True)


def create_dashboard_header(spot_price, vix_current, flip_strike):
    """Create the dashboard header with key metrics"""
    return dbc.Row([
        dbc.Col(html.H5(
            f"📊 SPX: {spot_price:.2f} | VIX: {vix_current:.2f} | Flip: {flip_strike:.1f}",
            className="text-center"
        ), width=12)
    ], className="mb-3")


def create_vix_momentum_row(vix_chart, momentum_chart):
    """Create the VIX analysis and momentum charts row"""
    return dbc.Row([
        dbc.Col(dcc.Graph(figure=vix_chart, style={"height": "400px"}), width=6),
        dbc.Col(dcc.Graph(figure=momentum_chart, style={"height": "400px"}), width=6)
    ], className="mb-4")


def create_targets_row(targets_chart):
    """Create the targets analysis row"""
    return dbc.Row([
        dbc.Col(dcc.Graph(figure=targets_chart, style={"height": "500px"}), width=12)
    ], className="mb-4")


def create_gex_charts_row(gex_chart, dex_chart, oi_chart, smile_chart):
    """Create the main GEX/DEX/OI/Smile charts row"""
    return dbc.Row([
        dbc.Col(dcc.Graph(figure=gex_chart, style={"height": "600px"}), width=3),
        dbc.Col(dcc.Graph(figure=dex_chart, style={"height": "600px"}), width=3),
        dbc.Col(dcc.Graph(figure=oi_chart, style={"height": "600px"}), width=3),
        dbc.Col(dcc.Graph(figure=smile_chart, style={"height": "600px"}), width=3)
    ])


def create_dashboard_content(charts, spot_price, vix_data, flip_strike):
    """Create the complete dashboard content layout"""
    return html.Div([
        # Header with key metrics
        create_dashboard_header(spot_price, vix_data['current'], flip_strike),

        # VIX Analysis and Flow Momentum Row
        create_vix_momentum_row(charts['vix_analysis'], charts['flow_momentum']),

        # Target Analysis Row
        create_targets_row(charts['targets']),

        # Original GEX/DEX/OI/Smile charts
        create_gex_charts_row(
            charts['gex'],
            charts['dex'],
            charts['oi'],
            charts['smile']
        )
    ])


def create_error_layout(error_message):
    """Create error display layout"""
    return html.Div([
        dbc.Alert(
            [
                html.H4("⚠️ Error", className="alert-heading"),
                html.P(f"Error updating dashboard: {error_message}"),
                html.Hr(),
                html.P("Please check your connection and try again.", className="mb-0"),
            ],
            color="danger",
        )
    ])


def create_login_required_layout():
    """Create layout shown when user needs to log in"""
    return html.Div([
        dbc.Alert(
            [
                html.H4("🔑 Login Required", className="alert-heading"),
                html.P("Please log in to view the dashboard."),
            ],
            color="info",
        )
    ])


def create_loading_layout():
    """Create loading state layout"""
    return html.Div([
        dbc.Spinner(
            html.Div([
                html.H5("📊 Loading market data..."),
                html.P("Fetching options chains, calculating GEX levels, and analyzing flows.")
            ]),
            color="primary",
            type="border"
        )
    ], className="text-center p-5")