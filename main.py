"""
Main application entry point for SPX Ultimate Trading Dashboard
"""

import flask
import nest_asyncio
from dash import Dash
import dash_bootstrap_components as dbc

from config import (
    APP_TITLE, APP_PORT, APP_DEBUG,
    ensure_directories
)
from layout import create_layout
from callbacks import register_callbacks

# Global data storage for momentum calculations
flow_history = []
vix_history = []
target_history = []


def create_app():
    """Create and configure the Dash application"""

    # Create Flask server
    server = flask.Flask(__name__)

    # Create Dash app
    app = Dash(
        __name__,
        server=server,
        external_stylesheets=[dbc.themes.DARKLY]
    )
    app.title = APP_TITLE

    # Set layout
    app.layout = create_layout()

    # Register callbacks
    register_callbacks(app)

    return app


def main():
    """Main application entry point"""

    # Apply nest_asyncio for async operations
    nest_asyncio.apply()

    # Ensure required directories exist
    ensure_directories()

    # Create the app
    app = create_app()

    # Print startup information
    print("🚀 Starting SPX Ultimate Trading Dashboard...")
    print("📊 Features: VIX Analysis, Flow Momentum, Dynamic Targets, PineScript Export")
    print(f"🌐 Running on http://localhost:{APP_PORT}")

    # Run the application
    app.run_server(
        debug=APP_DEBUG,
        use_reloader=False,
        port=APP_PORT,
        host='0.0.0.0'  # Allow external connections
    )


if __name__ == '__main__':
    main()