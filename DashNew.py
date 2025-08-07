import asyncio
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from tastytrade import Session, DXLinkStreamer
from tastytrade.dxfeed import Greeks, Summary, TimeAndSale
from tastytrade.market_data import get_market_data
from tastytrade.utils import today_in_new_york, is_market_open_on
from tastytrade.instruments import NestedOptionChain
from tastytrade.order import InstrumentType
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import flask
import os
import plotly.io as pio
import nest_asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback
# from scipy import stats
# from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

pio.templates.default = "plotly_dark"

# === SETTINGS ===
TIMEOUT_SECONDS = 15
REFRESH_INTERVAL = 45 * 1000  # milliseconds
PINE_FILE = "generated/gex_levels.pine"

# Flow momentum lookback periods (in refresh intervals)
MOMENTUM_PERIODS = [3, 6, 12, 24]  # 3min, 6min, 12min, 24min with 45s refresh

# === DASH APP ===
server = flask.Flask(__name__)
app = Dash(__name__, server=server, external_stylesheets=[dbc.themes.DARKLY])
app.title = "SPX Ultimate Trading Dashboard"

# Global data storage for momentum calculations and logs
flow_history = []
vix_history = []
target_history = []
log_messages = []  # New global log storage


def add_log_message(level, message):
    """Add a timestamped log message"""
    global log_messages
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}"
    log_messages.append(log_entry)
    log_messages = log_messages[-100:]  # Keep last 100 messages
    print(log_entry)  # Also print to console


# === ENHANCED LAYOUT ===
app.layout = dbc.Container([
    html.H1("🎯 SPX Ultimate Trading Dashboard - VIX + Flow + Targets", className="text-center my-4"),
    dcc.Store(id='session-data', storage_type='session'),
    dbc.Row([
        dbc.Col([
            html.H4("🔑 Login to Tastytrade"),
            html.Label("Username:"),
            dbc.Input(id='email', placeholder="Email", type="email", className="mb-2"),
            html.Label("Password:"),
            dbc.Input(id='password', placeholder="Password", type="password", className="mb-2"),
            dbc.Button("Login", id='login-btn', color="primary", className="mb-3 w-100"),
            dbc.Alert(id='login-alert', is_open=False, duration=4000),

            html.Hr(),
            html.H5("⚙️ Settings"),
            html.Label("Symbol (SPX, SPY, ESU5):"),
            dbc.Input(id='symbol', placeholder="Symbol (e.g., SPX)", value="SPX", className="mb-2"),
            html.Label("Strike Range (+/-):"),
            dbc.Input(id='strike_range', placeholder="Strike Range", type="number", value=120, className="mb-2"),
            html.Label("DTE Limit (days):"),
            dbc.Input(id='dte_limit', placeholder="DTE Limit", type="number", value=1, className="mb-2"),
            html.Label("Hedging Radius (pts):"),
            dbc.Input(id='hedge_radius', placeholder="Radius (e.g., 50)", type="number", value=50, className="mb-2"),
            html.Label("Decay Factor (0.01-0.1):"),
            dbc.Input(id='decay_factor', placeholder="Decay (e.g., 0.05)", type="number", value=0.05, step=0.01,
                      className="mb-2"),
            html.Label("Target Confidence Threshold:"),
            dbc.Input(id='confidence_threshold', placeholder="Confidence %", type="number", value=70, className="mb-3"),

            # Weight controls
            html.Label("Gamma Weight (w1):"),
            dbc.Input(id='w1', placeholder="Gamma weight", type="number", value=1.0, step=0.1, className="mb-2"),
            html.Label("Delta Weight (w2):"),
            dbc.Input(id='w2', placeholder="Delta weight", type="number", value=1.0, step=0.1, className="mb-2"),
            html.Label("Charm Weight (w3):"),
            dbc.Input(id='w3', placeholder="Charm weight", type="number", value=1.0, step=0.1, className="mb-2"),
            html.Label("Vega Weight (w4):"),
            dbc.Input(id='w4', placeholder="Vega weight", type="number", value=0.5, step=0.1, className="mb-2"),
            html.Label("VIX Weight (w5):"),
            dbc.Input(id='w5', placeholder="VIX weight", type="number", value=0.8, step=0.1, className="mb-3"),

            dbc.Button("🔄 Update Settings", id='update-settings-btn', color="secondary", className="w-100 mb-3"),

            # Signal Display
            html.Hr(),
            html.H5("🎯 Live Signals"),
            html.Div(id='signal-display', className="mb-3"),

            # Log Output Box - NEW SECTION
            html.Hr(),
            html.H5("📋 System Log"),
            html.Div([
                dbc.Button("🗑️ Clear Log", id='clear-log-btn', color="outline-secondary", size="sm", className="mb-2"),
                dcc.Textarea(
                    id='log-output',
                    style={
                        "width": "100%",
                        "height": "300px",
                        "backgroundColor": "#2b2b2b",
                        "color": "#ffffff",
                        "fontFamily": "monospace",
                        "fontSize": "12px",
                        "border": "1px solid #444",
                        "padding": "10px"
                    },
                    readOnly=True,
                    value="System initialized. Waiting for data...\n"
                )
            ]),

            html.Hr(),
            dbc.Button("📋 Generate PineScript", id="copy-pine-btn", color="secondary", className="w-100 mb-2"),
            dcc.Textarea(id="pine-textarea", style={"width": "100%", "height": 200}, readOnly=True, value=""),
            dcc.Clipboard(id="clipboard", target_id="pine-textarea", title="Copy Pinescript", style={
                "display": "inline-block", "fontSize": 24, "cursor": "pointer", "marginTop": "5px"
            }),
        ], width=3),
        dbc.Col([
            dcc.Loading(
                id="loading-dashboard",
                type="circle",
                children=html.Div(id='dashboard-content')
            ),
            dcc.Interval(id='interval-component', interval=REFRESH_INTERVAL, n_intervals=0)
        ], width=9)
    ])
], fluid=True)


# === ENHANCED CALLBACKS ===

@app.callback(
    Output('session-data', 'data'),
    Output('login-alert', 'children'),
    Output('login-alert', 'color'),
    Output('login-alert', 'is_open'),
    Input('login-btn', 'n_clicks'),
    State('email', 'value'),
    State('password', 'value'),
    prevent_initial_call=True
)
def login(n_clicks, email, password):
    try:
        session = Session(email, password)
        add_log_message("INFO", f"Successfully logged in as {email}")
        return {
            "email": email,
            "password": password,
            "logged_in": True
        }, "✅ Login successful!", "success", True
    except Exception as e:
        add_log_message("ERROR", f"Login failed: {e}")
        return {}, f"❌ Login failed: {e}", "danger", True


@app.callback(
    Output('log-output', 'value'),
    Input('clear-log-btn', 'n_clicks'),
    Input('interval-component', 'n_intervals'),
    prevent_initial_call=False
)
def update_log_output(clear_clicks, n_intervals):
    """Update the log output display"""
    global log_messages

    # Check if clear button was clicked
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'clear-log-btn.n_clicks':
        log_messages = []
        add_log_message("INFO", "Log cleared by user")

    # Return all log messages joined with newlines
    return "\n".join(log_messages) + "\n"


@app.callback(
    [Output('dashboard-content', 'children'),
     Output('signal-display', 'children')],
    [Input('session-data', 'data'),
     Input('interval-component', 'n_intervals'),
     Input('update-settings-btn', 'n_clicks')],
    [State('symbol', 'value'),
     State('strike_range', 'value'),
     State('dte_limit', 'value'),
     State('hedge_radius', 'value'),
     State('decay_factor', 'value'),
     State('w1', 'value'),
     State('w2', 'value'),
     State('w3', 'value'),
     State('w4', 'value'),
     State('w5', 'value'),
     State('confidence_threshold', 'value')],
    prevent_initial_call=True
)
def update_dashboard(session_data, n_intervals, update_clicks, symbol, strike_range,
                     dte_limit, hedge_radius, decay_factor, w1, w2, w3, w4, w5, confidence_threshold):
    if not session_data or not session_data.get('logged_in'):
        return html.Div("Please log in to view the dashboard."), html.Div("")

    try:
        add_log_message("INFO", f"Updating dashboard for {symbol} (interval #{n_intervals})")

        # Fetch enhanced data including VIX
        enhanced_data = fetch_enhanced_market_data(
            session_data['email'], session_data['password'], symbol.upper(),
            strike_range, dte_limit
        )

        gex_df = enhanced_data['gex_df']
        spot_price = enhanced_data['spot_price']
        vix_data = enhanced_data['vix_data']
        next_exp = enhanced_data['next_exp']
        flip_strike = enhanced_data['flip_strike']

        add_log_message("INFO",
                        f"Fetched data: Spot={spot_price:.2f}, VIX={vix_data['current']:.2f}, Flip={flip_strike:.1f}")

        # Store current data for momentum calculations
        current_timestamp = datetime.now()
        store_historical_data(gex_df, vix_data, spot_price, current_timestamp)

        # Calculate flow momentum
        momentum_data = calculate_flow_momentum(gex_df, spot_price)
        add_log_message("INFO",
                        f"Flow momentum: {momentum_data['momentum_strength']}, GEX_short={momentum_data['gex_momentum_short']:.0f}")

        # Calculate VIX analysis
        vix_analysis = analyze_vix_regime(vix_data, spot_price)
        add_log_message("INFO",
                        f"VIX regime: {vix_analysis['regime']}, Term structure: {vix_analysis['term_structure']}")

        # Generate targets with confidence scores
        targets = calculate_dynamic_targets(
            gex_df, spot_price, vix_analysis, momentum_data,
            hedge_radius, decay_factor, (w1, w2, w3, w4, w5)
        )

        # Store targets in global history and log them
        global target_history
        if targets:
            target_history.append(targets)
            target_history = target_history[-10:]  # Keep last 10 target sets

            # Log top 3 targets
            top_targets = targets[:3]
            for i, target in enumerate(top_targets, 1):
                add_log_message("TARGET",
                                f"#{i}: {target['strike']:.0f} ({target['direction']}) - "
                                f"{target['confidence']:.0f}% conf, {target['gex_type']}, "
                                f"ETA: {target['expected_time_hours']:.1f}h")

        # Build enhanced charts
        charts = build_enhanced_charts(gex_df, spot_price, vix_analysis, momentum_data)

        # Generate trading signals
        signals = generate_trading_signals(targets, momentum_data, vix_analysis, confidence_threshold)

        # Log all signals
        for signal in signals:
            signal_level = "SIGNAL" if signal.get('confidence', 0) >= confidence_threshold else "WEAK_SIGNAL"
            add_log_message(signal_level,
                            f"{signal.get('signal_type', 'UNKNOWN')}: {signal.get('message', 'No message')} "
                            f"[Action: {signal.get('action', 'N/A')}]")

        # Build dashboard layout
        dashboard_content = html.Div([
            # Header with key metrics
            dbc.Row([
                dbc.Col(html.H5(f"📊 SPX: {spot_price:.2f} | VIX: {vix_data['current']:.2f} | Flip: {flip_strike:.1f}",
                                className="text-center"), width=12)
            ], className="mb-3"),

            # VIX Analysis Row
            dbc.Row([
                dbc.Col(dcc.Graph(figure=charts['vix_analysis'], style={"height": "400px"}), width=6),
                dbc.Col(dcc.Graph(figure=charts['flow_momentum'], style={"height": "400px"}), width=6)
            ], className="mb-4"),

            # Target Analysis Row
            dbc.Row([
                dbc.Col(dcc.Graph(figure=charts['targets'], style={"height": "500px"}), width=12)
            ], className="mb-4"),

            # Original GEX/DEX/OI charts
            dbc.Row([
                dbc.Col(dcc.Graph(figure=charts['gex'], style={"height": "600px"}), width=3),
                dbc.Col(dcc.Graph(figure=charts['dex'], style={"height": "600px"}), width=3),
                dbc.Col(dcc.Graph(figure=charts['oi'], style={"height": "600px"}), width=3),
                dbc.Col(dcc.Graph(figure=charts['smile'], style={"height": "600px"}), width=3)
            ])
        ])

        # Generate signal display
        signal_display = build_signal_display(signals, targets)

        add_log_message("INFO", f"Dashboard update completed successfully")

        return dashboard_content, signal_display

    except Exception as e:
        error_msg = f"Error updating dashboard: {e}"
        add_log_message("ERROR", error_msg)
        print("⚠ Full exception traceback:")
        traceback.print_exc()
        return html.Div(f"⚠ {error_msg}"), html.Div("")


# === ENHANCED CORE FUNCTIONS ===

def fetch_enhanced_market_data(email, password, symbol, strike_range, dte_limit):
    """Fetch SPX options data + VIX analysis"""
    session = Session(email, password)

    # Get SPX data
    mdata = get_market_data(session, symbol, InstrumentType.INDEX)
    spot_price = float(mdata.last)

    # Get VIX data
    try:
        vix_mdata = get_market_data(session, "VIX", InstrumentType.INDEX)
        vix_current = float(vix_mdata.last)

        # Try to get VIX9D and VIX3M for term structure
        try:
            vix9d_mdata = get_market_data(session, "VIX9D", InstrumentType.INDEX)
            vix9d = float(vix9d_mdata.last)
        except:
            vix9d = vix_current * 0.95  # Approximate if not available

        try:
            vix3m_mdata = get_market_data(session, "VIX3M", InstrumentType.INDEX)
            vix3m = float(vix3m_mdata.last)
        except:
            vix3m = vix_current * 1.05  # Approximate if not available

    except:
        # Fallback if VIX not available
        vix_current = 20.0
        vix9d = 19.0
        vix3m = 21.0

    vix_data = {
        'current': vix_current,
        'vix9d': vix9d,
        'vix3m': vix3m,
        'term_structure': vix3m / vix_current,
        'short_term_premium': vix_current / vix9d
    }

    # Get options chain (existing logic)
    next_exp = today_in_new_york()
    while not is_market_open_on(next_exp):
        next_exp += timedelta(days=1)

    chain = NestedOptionChain.get(session, symbol)
    subs_list, filtered_options = [], []

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

    add_log_message("INFO", f"Processing {len(filtered_options)} option strikes")

    # Get Greeks and Summary data
    greeks_data, summary_data = run_enhanced_streams(session, subs_list)

    # Process into GEX DataFrame (existing logic)
    gex_df = process_options_data(greeks_data, summary_data, filtered_options, dte_limit)

    # Calculate flip strike
    flip_idx = gex_df['Cumulative_GEX'].abs().idxmin() if not gex_df.empty else 0
    flip_strike = gex_df.loc[flip_idx, 'strike'] if not gex_df.empty else spot_price

    return {
        'gex_df': gex_df,
        'spot_price': spot_price,
        'vix_data': vix_data,
        'next_exp': next_exp.strftime('%Y-%m-%d'),
        'flip_strike': flip_strike
    }


def run_enhanced_streams(session, subs_list):
    """Enhanced streaming with better error handling"""
    executor = ThreadPoolExecutor(max_workers=2)
    return executor.submit(run_streams_sync, session, subs_list).result()


def run_streams_sync(session, subs_list):
    """Synchronous wrapper for async streaming"""
    greeks_data, summary_data = {}, {}

    async def collect():
        async with DXLinkStreamer(session) as streamer:
            await streamer.subscribe(Greeks, subs_list)
            await streamer.subscribe(Summary, subs_list)
            await asyncio.wait_for(asyncio.gather(
                collect_greeks(streamer, greeks_data, subs_list),
                collect_summary(streamer, summary_data, subs_list)
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


async def collect_greeks(streamer, greeks_data, subs_list):
    """Collect Greeks data"""
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


async def collect_summary(streamer, summary_data, subs_list):
    """Collect Summary data"""
    async for event in streamer.listen(Summary):
        summary_data[event.event_symbol] = {
            "symbol": event.event_symbol,
            "open_interest": float(event.open_interest or 0)
        }
        if len(summary_data) >= len(subs_list):
            break


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

    # Keep only last 100 entries (about 1 hour of data)
    flow_history = flow_history[-100:]
    vix_history = vix_history[-100:]


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
    if short_strength > 1000:
        strength = 'STRONG'
    elif short_strength > 500:
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
    if current_vix < 15:
        regime = 'LOW_VOL'
        regime_color = 'green'
    elif current_vix < 25:
        regime = 'NORMAL_VOL'
        regime_color = 'yellow'
    elif current_vix < 35:
        regime = 'HIGH_VOL'
        regime_color = 'orange'
    else:
        regime = 'PANIC_VOL'
        regime_color = 'red'

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
    return sorted(targets, key=lambda x: x['confidence'], reverse=True)


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

    return signals


def build_signal_display(signals, targets):
    """Build the signal display component"""
    if not signals:
        return html.Div("No signals generated", className="text-muted")

    signal_cards = []

    for signal in signals:
        # Color coding
        if 'BULLISH' in signal.get('signal_type', ''):
            color = 'success'
            icon = '📈'
        elif 'BEARISH' in signal.get('signal_type', ''):
            color = 'danger'
            icon = '📉'
        elif 'WARNING' in signal.get('signal_type', ''):
            color = 'warning'
            icon = '⚠️'
        elif 'HEDGE' in signal.get('signal_type', ''):
            color = 'info'
            icon = '🛡️'
        else:
            color = 'secondary'
            icon = '⏸️'

        # Build card content
        card_content = [
            html.H6(f"{icon} {signal.get('signal_type', 'UNKNOWN')}", className=f"text-{color}"),
            html.P(signal.get('message', ''), className="mb-1"),
            html.Small(f"Action: {signal.get('action', 'N/A')}", className="text-muted")
        ]

        if signal.get('target_strike'):
            card_content.insert(-1, html.P(f"Target: {signal['target_strike']:.0f}", className="mb-1 fw-bold"))

        signal_cards.append(
            dbc.Card(
                dbc.CardBody(card_content),
                color=color,
                outline=True,
                className="mb-2"
            )
        )

    return html.Div(signal_cards)


def build_enhanced_charts(gex_df, spot_price, vix_analysis, momentum_data):
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
    fig_gex = go.Figure()
    fig_gex.add_trace(go.Bar(
        x=gex_df['strike'],
        y=gex_df['Net_GEX'],
        name='Net GEX',
        marker_color=['red' if x < 0 else 'green' for x in gex_df['Net_GEX']]
    ))
    fig_gex.add_vline(x=spot_price, line_dash="dash", line_color="white", annotation_text=f"Spot: {spot_price:.0f}")
    fig_gex.update_layout(
        title="Gamma Exposure (GEX)",
        xaxis_title="Strike Price",
        yaxis_title="Net GEX",
        template="plotly_dark",
        height=600
    )
    charts['gex'] = fig_gex

    # 2. DEX Chart
    fig_dex = go.Figure()
    fig_dex.add_trace(go.Bar(
        x=gex_df['strike'],
        y=gex_df['Net_DEX'],
        name='Net DEX',
        marker_color='blue'
    ))
    fig_dex.add_vline(x=spot_price, line_dash="dash", line_color="white", annotation_text=f"Spot: {spot_price:.0f}")
    fig_dex.update_layout(
        title="Delta Exposure (DEX)",
        xaxis_title="Strike Price",
        yaxis_title="Net DEX",
        template="plotly_dark",
        height=600
    )
    charts['dex'] = fig_dex

    # 3. Open Interest Chart
    fig_oi = go.Figure()
    fig_oi.add_trace(go.Bar(
        x=gex_df['strike'],
        y=gex_df['Call_OI'],
        name='Call OI',
        marker_color='green',
        opacity=0.7
    ))
    fig_oi.add_trace(go.Bar(
        x=gex_df['strike'],
        y=-gex_df['Put_OI'],
        name='Put OI',
        marker_color='red',
        opacity=0.7
    ))
    fig_oi.add_vline(x=spot_price, line_dash="dash", line_color="white", annotation_text=f"Spot: {spot_price:.0f}")
    fig_oi.update_layout(
        title="Open Interest",
        xaxis_title="Strike Price",
        yaxis_title="Open Interest",
        template="plotly_dark",
        height=600,
        barmode='relative'
    )
    charts['oi'] = fig_oi

    # 4. Volatility Smile
    fig_smile = go.Figure()
    if 'Call_IV' in gex_df.columns and 'Put_IV' in gex_df.columns:
        fig_smile.add_trace(go.Scatter(
            x=gex_df['strike'],
            y=gex_df['Call_IV'],
            mode='lines+markers',
            name='Call IV',
            line_color='green'
        ))
        fig_smile.add_trace(go.Scatter(
            x=gex_df['strike'],
            y=gex_df['Put_IV'],
            mode='lines+markers',
            name='Put IV',
            line_color='red'
        ))
        fig_smile.add_vline(x=spot_price, line_dash="dash", line_color="white",
                            annotation_text=f"Spot: {spot_price:.0f}")
    fig_smile.update_layout(
        title="Volatility Smile",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility",
        template="plotly_dark",
        height=600
    )
    charts['smile'] = fig_smile

    # 5. VIX Analysis Chart
    fig_vix = go.Figure()

    # VIX level with regime coloring
    regime_colors = {
        'LOW_VOL': 'green',
        'NORMAL_VOL': 'yellow',
        'HIGH_VOL': 'orange',
        'PANIC_VOL': 'red'
    }

    fig_vix.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=vix_analysis['current_vix'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"VIX ({vix_analysis['regime']})"},
        delta={'reference': 20},
        gauge={
            'axis': {'range': [None, 50]},
            'bar': {'color': regime_colors.get(vix_analysis['regime'], 'gray')},
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

    fig_vix.update_layout(
        template="plotly_dark",
        height=400,
        title=f"VIX Regime Analysis - {vix_analysis['term_structure']}"
    )
    charts['vix_analysis'] = fig_vix

    # 6. Flow Momentum Chart
    fig_momentum = go.Figure()

    if momentum_data and len(flow_history) > 0:
        df_flow = pd.DataFrame(flow_history)

        fig_momentum.add_trace(go.Scatter(
            x=df_flow['timestamp'],
            y=df_flow['total_gex'],
            mode='lines',
            name='Total GEX',
            line_color='green'
        ))

        fig_momentum.add_trace(go.Scatter(
            x=df_flow['timestamp'],
            y=df_flow['total_dex'],
            mode='lines',
            name='Total DEX',
            line_color='blue',
            yaxis='y2'
        ))

        # Add momentum annotations
        fig_momentum.add_annotation(
            x=df_flow['timestamp'].iloc[-1],
            y=df_flow['total_gex'].iloc[-1],
            text=f"Strength: {momentum_data['momentum_strength']}",
            showarrow=True,
            arrowhead=2
        )

    fig_momentum.update_layout(
        title=f"Flow Momentum ({momentum_data.get('momentum_strength', 'UNKNOWN')})",
        xaxis_title="Time",
        yaxis_title="GEX Flow",
        yaxis2=dict(
            title="DEX Flow",
            overlaying='y',
            side='right'
        ),
        template="plotly_dark",
        height=400
    )
    charts['flow_momentum'] = fig_momentum

    # 7. Dynamic Targets Chart
    fig_targets = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price Targets with Confidence', 'Target Timeline'),
        vertical_spacing=0.1
    )

    if len(target_history) > 0:
        # Plot targets as horizontal lines with confidence-based opacity
        for target in target_history[-1]:  # Most recent targets
            opacity = target['confidence'] / 100
            color = 'green' if target['direction'] == 'UPSIDE' else 'red'

            fig_targets.add_hline(
                y=target['strike'],
                line_dash="dash",
                line_color=color,
                opacity=opacity,
                annotation_text=f"{target['strike']:.0f} ({target['confidence']:.0f}%)",
                row=1, col=1
            )

    # Current price line
    fig_targets.add_hline(
        y=spot_price,
        line_color="white",
        line_width=3,
        annotation_text=f"Current: {spot_price:.0f}",
        row=1, col=1
    )

    fig_targets.update_layout(
        template="plotly_dark",
        height=500,
        showlegend=False
    )
    charts['targets'] = fig_targets

    return charts


@app.callback(
    [Output('pine-textarea', 'value')],
    [Input('copy-pine-btn', 'n_clicks')],
    [State('session-data', 'data'),
     State('symbol', 'value')],
    prevent_initial_call=True
)
def generate_pinescript(n_clicks, session_data, symbol):
    """Generate PineScript for TradingView"""
    if not session_data or not session_data.get('logged_in'):
        return ["Please log in first"]

    try:
        # Get latest GEX data for key levels
        enhanced_data = fetch_enhanced_market_data(
            session_data['email'], session_data['password'], symbol.upper(),
            120, 1  # Default parameters
        )

        gex_df = enhanced_data['gex_df']
        spot_price = enhanced_data['spot_price']

        if gex_df.empty:
            return ["// No data available for PineScript generation"]

        # Find key support/resistance levels
        gex_df['abs_gex'] = abs(gex_df['Net_GEX'])
        key_levels = gex_df.nlargest(8, 'abs_gex')[['strike', 'Net_GEX']].copy()

        # Generate PineScript
        pine_script = f'''// SPX GEX Levels - Auto-generated
//@version=5
indicator("SPX GEX Levels", overlay=true, max_lines_count=20, max_labels_count=20)

// Current spot price reference
current_price = {spot_price:.2f}

// Key GEX Levels
'''

        for i, (_, level) in enumerate(key_levels.iterrows()):
            strike = level['strike']
            gex_value = level['Net_GEX']
            level_type = "resistance" if gex_value > 0 else "support"
            color = "color.red" if gex_value > 0 else "color.green"

            pine_script += f'''
// Level {i + 1}: {strike:.0f} ({level_type})
level_{i + 1} = {strike:.2f}
line_{i + 1} = line.new(bar_index - 100, level_{i + 1}, bar_index + 100, level_{i + 1}, 
             color={color}, width=2, style=line.style_dashed)
label_{i + 1} = label.new(bar_index, level_{i + 1}, 
              text="{strike:.0f} ({level_type})", 
              color={color}, textcolor=color.white, size=size.small)
'''

        pine_script += f'''
// Current price level
current_line = line.new(bar_index - 100, current_price, bar_index + 100, current_price, 
                       color=color.white, width=3, style=line.style_solid)
current_label = label.new(bar_index, current_price, 
                         text="Current: {spot_price:.0f}", 
                         color=color.white, textcolor=color.black, size=size.normal)

// Alert conditions
alertcondition(close >= level_1, title="Approaching Key Resistance", message="Price approaching major resistance level")
alertcondition(close <= level_1, title="Approaching Key Support", message="Price approaching major support level")
'''

        return [pine_script]

    except Exception as e:
        return [f"// Error generating PineScript: {e}"]


# Add this global variable to store target history
target_history = []


# Update the callback to store target history
def update_dashboard_with_targets(session_data, n_intervals, update_clicks, symbol, strike_range,
                                  dte_limit, hedge_radius, decay_factor, w1, w2, w3, w4, w5, confidence_threshold):
    """Updated dashboard function that stores target history"""
    global target_history

    # ... existing code ...

    # After calculating targets
    targets = calculate_dynamic_targets(
        gex_df, spot_price, vix_analysis, momentum_data,
        hedge_radius, decay_factor, (w1, w2, w3, w4, w5)
    )

    # Store targets for PineScript and historical analysis
    if targets:
        target_history.append(targets)
        target_history = target_history[-10:]  # Keep last 10 target sets

    # ... rest of existing code ...


if __name__ == '__main__':
    nest_asyncio.apply()

    # Ensure directories exist
    os.makedirs('generated', exist_ok=True)

    print("🚀 Starting SPX Ultimate Trading Dashboard...")
    print("📊 Features: VIX Analysis, Flow Momentum, Dynamic Targets, PineScript Export")

    app.run(debug=True, use_reloader=False, port=8057)
