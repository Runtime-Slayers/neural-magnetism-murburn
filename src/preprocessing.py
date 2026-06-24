"""
Configuration preprocessing, yaml configuration loader, and matplotlib style initialization.
"""

import os
import yaml
import matplotlib

# Determine the configurations file path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(os.path.dirname(_CURRENT_DIR), 'configs', 'hyperparams.yaml')

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file not found at: {CONFIG_PATH}")

with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# Extract shared keys
COLORS = CONFIG['colors']
MATPLOTLIB_STYLES = CONFIG['matplotlib']
BIO_CONSTANTS = CONFIG['bio_constants']
HODGKIN_HUXLEY_PARAMS = CONFIG['hodgkin_huxley']
KURAMOTO_PARAMS = CONFIG['kuramoto']
ENERGY_PARAMS = CONFIG['energy_harvesting']

# Apply styling globally
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update(MATPLOTLIB_STYLES)

def get_color_palette():
    """Returns the HSL color palette dictionary."""
    return COLORS
