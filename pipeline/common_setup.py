# common_setup.py
# ================================
# Shared Setup: Libraries, Style, and Data Preparation
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle   
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import shapiro, jarque_bera, norm
from statsmodels.stats.diagnostic import acorr_ljungbox

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import time series analysis libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    print("All time series libraries imported successfully")
except ImportError as e:
    print("Missing required libraries. Please install with:")
    print("  pip install statsmodels scikit-learn")
    print(f"Error: {e}")
    exit(1)

# Set matplotlib style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('default')

# ================================
# Load and Prepare Data
# ================================
print("=== Loading and preparing Bitcoin data ===")

# Load data
df = pd.read_csv('data/bitcon_index_data2025.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%Y')
df.set_index('Date', inplace=True)
bitcoin_ts = df['Bitcoin']

# Create log-differenced series
bitcoin_log = np.log(bitcoin_ts)
bitcoin_log_diff = bitcoin_log.diff().dropna()
bitcoin_log_diff.index.freq = 'MS'

print(f"Original data shape: {bitcoin_ts.shape}")
print(f"Log-differenced data shape: {bitcoin_log_diff.shape}")
