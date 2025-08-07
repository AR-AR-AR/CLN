"""
Configuration settings and constants for SPX Trading Dashboard
"""

import os
import plotly.io as pio
import warnings

warnings.filterwarnings('ignore')

# Set default plotly template
pio.templates.default = "plotly_dark"

# === TRADING SETTINGS ===
TIMEOUT_SECONDS = 15
REFRESH_INTERVAL = 45 * 1000  # milliseconds
PINE_FILE = "generated/gex_levels.pine"

# Flow momentum lookback periods (in refresh intervals)
MOMENTUM_PERIODS = [3, 6, 12, 24]  # 3min, 6min, 12min, 24min with 45s refresh

# === DEFAULT PARAMETERS ===
DEFAULT_SYMBOL = "SPX"
DEFAULT_STRIKE_RANGE = 120
DEFAULT_DTE_LIMIT = 1
DEFAULT_HEDGE_RADIUS = 50
DEFAULT_DECAY_FACTOR = 0.05
DEFAULT_CONFIDENCE_THRESHOLD = 70

# Default weights for dynamic target calculation
DEFAULT_WEIGHTS = {
    'w1': 1.0,  # Gamma weight
    'w2': 1.0,  # Delta weight
    'w3': 1.0,  # Charm weight
    'w4': 0.5,  # Vega weight
    'w5': 0.8   # VIX weight
}

# === VIX REGIME THRESHOLDS ===
VIX_THRESHOLDS = {
    'LOW_VOL': 15,
    'NORMAL_VOL': 25,
    'HIGH_VOL': 35
}

# VIX regime colors
VIX_REGIME_COLORS = {
    'LOW_VOL': 'green',
    'NORMAL_VOL': 'yellow',
    'HIGH_VOL': 'orange',
    'PANIC_VOL': 'red'
}

# === DATA STORAGE LIMITS ===
MAX_FLOW_HISTORY = 100  # Keep last 100 entries (about 1 hour of data)
MAX_VIX_HISTORY = 100
MAX_TARGET_HISTORY = 10

# === CHART SETTINGS ===
CHART_HEIGHT_MAIN = 600
CHART_HEIGHT_SECONDARY = 400
CHART_HEIGHT_TARGETS = 500

# === SIGNAL CLASSIFICATION ===
MOMENTUM_THRESHOLDS = {
    'STRONG': 1000,
    'MODERATE': 500
}

# === DIRECTORIES ===
GENERATED_DIR = "generated"

def ensure_directories():
    """Ensure required directories exist"""
    os.makedirs(GENERATED_DIR, exist_ok=True)

# === APP METADATA ===
APP_TITLE = "SPX Ultimate Trading Dashboard"
APP_PORT = 8057
APP_DEBUG = True