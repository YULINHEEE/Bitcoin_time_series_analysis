# Time Series Analysis Step 1: Stationarity Test and Data Transformation

# Data: Monthly historical performance of the Bitcoin index (in USD) 
#       between August 2011 and January 2025
# ================================
# improt packages
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, shapiro
import warnings
warnings.filterwarnings('ignore')

# Import time series analysis libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import itertools
    print("All time series libraries imported successfully")

except ImportError as e:
    print(f"Missing required libraries. Please install them using:")
    print("pip install statsmodels scikit-learn")
    print(f"Error details: {e}")
    exit(1)

# Set chart style
plt.style.use('seaborn-v0_8')

# ================================
# Data Loading and Preprocessing
# ================================

# Read data
df = pd.read_csv('data/bitcon_index_data2025.csv')
print(f"Data length: {len(df)}")

# Convert date column to datetime format
df['Date'] = pd.to_datetime(df.iloc[:, 0])
df.set_index('Date', inplace=True)

# Create time series object
bitcoin_ts = df.iloc[:, 0]
print(f"Data type: {type(bitcoin_ts)}")
print(f"Data head:\n{bitcoin_ts.head()}")
print(f"Basic statistics:\n{bitcoin_ts.describe()}")

# ================================
# Descriptive Statistical Analysis
# ================================

# Time series plot
plt.figure(figsize=(12, 6))
plt.plot(bitcoin_ts.index, bitcoin_ts.values, linewidth=1.5)
plt.title('Monthly Bitcoin Index Time Series')
plt.xlabel('Time')
plt.ylabel('Bitcoin Index (USD)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig('Figure 1. Bitcoin Time Series.png', dpi=300, bbox_inches='tight')
#plt.show()

# ================================
# Autocorrelation Analysis
# ================================

# Calculate first-order autocorrelation coefficient
def lag_correlation(series, lag=1):
    """Calculate lag correlation coefficient"""
    return series.corr(series.shift(lag))

first_order_corr = lag_correlation(bitcoin_ts, 1)
print(f"First-order autocorrelation coefficient: {first_order_corr:.4f}")

# Lag scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(bitcoin_ts[:-1], bitcoin_ts[1:], alpha=0.6)
plt.xlabel('Previous Bitcoin Index')
plt.ylabel('Current Bitcoin Index')
plt.title('Bitcoin Index Scatter Plot for Consecutive Months')
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig('Figure 2. Lag Scatter Plot.png', dpi=300, bbox_inches='tight')
#plt.show()

# ACF and PACF plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
plot_acf(bitcoin_ts, lags=42, ax=ax1, title='ACF Plot')
plot_pacf(bitcoin_ts, lags=42, ax=ax2, title='PACF Plot')
plt.tight_layout()
#plt.savefig('Figure 3. ACF_PACF_original.png', dpi=300, bbox_inches='tight')
#plt.show()

# ================================
# Stationarity Tests
# ================================

def stationarity_tests(series, name="Time Series"):
    """Perform multiple stationarity tests"""
    print(f"\n===== Stationarity Tests for {name} =====")
    
    # ADF Test
    adf_result = adfuller(series)
    print(f"=== ADF Test: ===")
    print(f"ADF Statistic: {adf_result[0]:.3f}")
    print(f"p-value: {adf_result[1]:.3f}")
    print(f"ADF_result: {adf_result}")
    print(f"Critical Values: {adf_result[4]}")
    if adf_result[1] <= 0.05:
        print("Result: Reject null hypothesis - Series is stationary")
    else:
        print("Result: Fail to reject null hypothesis - Series is non-stationary")
    
    # KPSS Test
    kpss_result = kpss(series)
    print(f"\n=== KPSS Test: ===")
    print(f"KPSS Statistic: {kpss_result[0]:.3f}")
    print(f"p-value: {kpss_result[1]:.3f}")
    print(f"kpss_result: {kpss_result}")
    print(f"Critical Values: {kpss_result[3]}")
    if kpss_result[1] <= 0.05:
        print("  Result: Reject null hypothesis - Series is non-stationary")
    else:
        print("  Result: Fail to reject null hypothesis - Series is stationary")

# Test original series
stationarity_tests(bitcoin_ts.dropna(), "Original Bitcoin Series")

# ================================
# Normality Tests
# ================================

def normality_tests(series, name="Time Series"):
    """Perform normality tests"""
    print(f"\n===== Normality Tests for {name} =====")
    
    # Shapiro-Wilk test
    if len(series) <= 5000:  # Shapiro-Wilk has sample size limitations
        shapiro_stat, shapiro_p = shapiro(series)
        print(f"Shapiro-Wilk Test:")
        print(f"  Statistic: {shapiro_stat:.3f}")
        print(f"  p-value: {shapiro_p:.3f}")
        if shapiro_p <= 0.05:
            print("  Result: Reject null hypothesis - Data is not normally distributed")
        else:
            print("  Result: Fail to reject null hypothesis - Data is normally distributed")
    
    # Jarque-Bera test
    jb_stat, jb_p = jarque_bera(series)
    print(f"\nJarque-Bera Test:")
    print(f"  Statistic: {jb_stat:.3f}")
    print(f"  p-value: {jb_p:.3f}")
    if jb_p <= 0.05:
        print("  Result: Reject null hypothesis - Data is not normally distributed")
    else:
        print("  Result: Fail to reject null hypothesis - Data is normally distributed")

# Test normality of original series
normality_tests(bitcoin_ts.dropna(), "Original Bitcoin Series")

# Q-Q plot
plt.figure(figsize=(10, 6))
stats.probplot(bitcoin_ts.dropna(), dist="norm", plot=plt)
plt.title('Figure 4. Q-Q Plot for Original Bitcoin Series')
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig('Figure 4. Q-Q plot Original.png', dpi=300, bbox_inches='tight')
#plt.show()

# ================================
# Data Transformation
# ================================

print("\n===== Data Transformation =====")

# Log transformation
bitcoin_log = np.log(bitcoin_ts)
print("Log transformation completed")

# Test stationarity of log-transformed series
stationarity_tests(bitcoin_log.dropna(), "Log-transformed Bitcoin Series")

# ACF and PACF plots for log-transformed series
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
plot_acf(bitcoin_log.dropna(), lags=42, ax=ax1, title='ACF Plot - Log Transformed')
plot_pacf(bitcoin_log.dropna(), lags=42, ax=ax2, title='PACF Plot - Log Transformed')
plt.tight_layout()
#plt.savefig('Figure 5. ACF_PACF_log.png', dpi=300, bbox_inches='tight')
#plt.show()

# First-order differencing
bitcoin_log_diff = bitcoin_log.diff().dropna()
print("First-order differencing completed")

# Test stationarity of differenced series
stationarity_tests(bitcoin_log_diff, "Log-differenced Bitcoin Series")

# Test normality of differenced series
normality_tests(bitcoin_log_diff, "Log-differenced Bitcoin Series")

# Plot differenced series
plt.figure(figsize=(12, 6))
plt.plot(bitcoin_log_diff.index, bitcoin_log_diff.values, linewidth=1.5)
plt.title('Figure 6. Log-differenced Bitcoin Series')
plt.xlabel('Time')
plt.ylabel('Log-differenced Bitcoin Index')
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig('Figure 6. Bitcoin log-diff.png', dpi=300, bbox_inches='tight')
#plt.show()

# ACF and PACF plots for differenced series
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
plot_acf(bitcoin_log_diff, lags=42, ax=ax1, title='ACF Plot - Log Differenced')
plot_pacf(bitcoin_log_diff, lags=42, ax=ax2, title='PACF Plot - Log Differenced')
plt.tight_layout()
#plt.savefig('Figure 7. ACF-PACF log-diff.png', dpi=300, bbox_inches='tight')
#plt.show()

# ================================
# Normality Tests for log-differenced series
# ================================

# Q-Q plot for differenced series
plt.figure(figsize=(10, 6))
stats.probplot(bitcoin_log_diff, dist="norm", plot=plt)
plt.title('Figure 8. Q-Q Plot for Log-differenced Bitcoin Series')
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig('Figure 8. qq_plot_log_diff.png', dpi=300, bbox_inches='tight')
#plt.show()

# Save the log-differenced series to a file
import pickle  # Add pickle import

SAVE_RESULTS = False

if SAVE_RESULTS:
    with open('bitcoin_log_diff.pkl', 'wb') as f:
        pickle.dump(bitcoin_log_diff, f)
    print("Log-differenced Bitcoin series saved to 'bitcoin_log_diff.pkl'")

    with open('bitcoin_log.pkl', 'wb') as f:
        pickle.dump(bitcoin_log, f) 
    print("Log Bitcoin series saved to 'bitcoin_log.pkl")

