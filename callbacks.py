"""
Dash callback functions for SPX Trading Dashboard
"""

import traceback
from datetime import datetime
from dash import Input, Output, State, ctx, html
import dash_bootstrap_components as dbc

from config import DEFAULT_CONFIDENCE_THRESHOLD
from auth import login_user
from data_fetcher import fetch_enhanced_market_data
from data_processor import store_historical_data
from analysis import calculate_flow_momentum, analyze_vix_regime, calculate_dynamic_targets
from charts import build_enhanced_charts
from signals import generate_trading_signals, build_signal_display
from pinescript import generate_pinescript_code

# Global data storage for momentum calculations
flow_history = []
vix_history = []
target_history = []

def register_callbacks(app):
    """Register all callbacks with the Dash app"""

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
    def handle_login(n_clicks, email, password):
        """Handle user login"""
        try:
            session_data = login_user(email, password)
            return session_data, "✅ Login successful!", "success", True
        except Exception as e:
            return {}, f"❌ Login failed: {e}", "danger", True

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
        """Main dashboard update callback"""
        if not session_data or not session_data.get('logged_in'):
            return html.Div("Please log in to view the dashboard."), html.Div("")

        try:
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

            # Store current data for momentum calculations
            current_timestamp = datetime.now()
            store_historical_data(gex_df, vix_data, spot_price, current_timestamp,
                                flow_history, vix_history)

            # Calculate flow momentum
            momentum_data = calculate_flow_momentum(gex_df, spot_price, flow_history)

            # Calculate VIX analysis
            vix_analysis = analyze_vix_regime(vix_data, spot_price, vix_history)

            # Generate targets with confidence scores
            targets = calculate_dynamic_targets(
                gex_df, spot_price, vix_analysis, momentum_data,
                hedge_radius, decay_factor, (w1, w2, w3, w4, w5)
            )

            # Store targets in global history
            global target_history
            if targets:
                target_history.append(targets)
                target_history = target_history[-10:]  # Keep last 10 target sets
                print(f"✅ Updated target_history with {len(targets)} targets")

            # Build enhanced charts
            charts = build_enhanced_charts(gex_df, spot_price, vix_analysis, momentum_data,
                                         flow_history, target_history)

            # Generate trading signals
            signals = generate_trading_signals(targets, momentum_data, vix_analysis,
                                             confidence_threshold or DEFAULT_CONFIDENCE_THRESHOLD)

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

            return dashboard_content, signal_display

        except Exception as e:
            print("⚠ Full exception traceback:")
            traceback.print_exc()
            return html.Div(f"⚠ Error updating dashboard: {e}"), html.Div("")

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

            # Generate PineScript
            pine_script = generate_pinescript_code(gex_df, spot_price)
            return [pine_script]

        except Exception as e:
            return [f"// Error generating PineScript: {e}"]