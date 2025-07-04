# Time Series Analysis Assignment 2 - Part 4: Residual Diagnostics and Forecasting

from common_setup import *

# Import additional libraries for residual diagnostics
from scipy.stats import jarque_bera, shapiro
from scipy import stats

# Try to import statsmodels diagnostic functions
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox # type: ignore
except ImportError:
    print("Warning: statsmodels diagnostic functions not available")
    
def simple_durbin_watson(residuals):
    """Simple Durbin-Watson implementation"""
    residuals = np.array(residuals)
    diff = np.diff(residuals)
    return np.sum(diff**2) / np.sum(residuals**2)

# import best model pkl
with open("pkl/best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

print("=" * 60)
print("RESIDUAL DIAGNOSTICS AND MODEL VALIDATION")
print("=" * 60)

# ================================
# Residual Diagnostics
# ================================

def comprehensive_residual_analysis(model, data, model_name="ARIMA Model"):
    """Comprehensive residual analysis"""
    
    print(f"\n{model_name} Residual Diagnostics Results")
    print("-" * 50)
    
    # Get residuals
    residuals = model.resid
    standardized_residuals = residuals / np.std(residuals)
    
    # Align indices for plotting - use residuals index directly
    residual_index = residuals.index
    
    # Create residual diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Residual time series plot
    axes[0, 0].plot(residual_index, residuals, linewidth=1)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_title(f'{model_name} Residual Time Series')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Standardized residual time series plot
    axes[0, 1].plot(residual_index, standardized_residuals, linewidth=1)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axhline(y=2, color='red', linestyle=':', alpha=0.5)
    axes[0, 1].axhline(y=-2, color='red', linestyle=':', alpha=0.5)
    axes[0, 1].set_title(f'{model_name} Standardized Residuals')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Standardized Residuals')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residual histogram
    axes[0, 2].hist(residuals, bins=30, alpha=0.7, edgecolor='black', density=True)
    axes[0, 2].set_title(f'{model_name} Residual Histogram')
    axes[0, 2].set_xlabel('Residuals')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add normal distribution curve
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 2].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 
                    'r-', linewidth=2, label='Normal Distribution')
    axes[0, 2].legend()
    
    # 4. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'{model_name} Residual Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Residual ACF plot
    plot_acf(residuals, lags=40, ax=axes[1, 1], title=f'{model_name} Residual ACF')
    
    # 6. Residual PACF plot
    plot_pacf(residuals, lags=40, ax=axes[1, 2], title=f'{model_name} Residual PACF')
    
    plt.tight_layout()
    #plt.savefig(f'Figure 12.{model_name.replace(" ", "_")} residual diagnostics.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    # ================================
    # Residual Statistical Tests
    # ================================
    
    print("\nResidual Statistical Test Results:")
    print("-" * 30)
    
    # 1. Normality tests
    print("1. Normality Tests:")
    if len(residuals) <= 5000:
        shapiro_stat, shapiro_p = shapiro(residuals)
        print(f"   Shapiro-Wilk Test: Statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
        print(f"   Normality Conclusion: {'Accept' if shapiro_p > 0.05 else 'Reject'} normality assumption")
    
    jb_result = jarque_bera(residuals)
    jb_stat = jb_result[0]
    jb_p_val = jb_result[1]
    print(f"   Jarque-Bera Test: Statistic={jb_stat:.4f}, p-value={jb_p_val}")
    print("   Normality Conclusion: Check p-value against 0.05 threshold")
    
    # 2. Autocorrelation test (Ljung-Box test)
    print("\n2. Residual Autocorrelation Test (Ljung-Box):")
    try:
        lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
        lb_stat = lb_result['lb_stat'].iloc[-1]
        lb_p = lb_result['lb_pvalue'].iloc[-1]
        print(f"   Ljung-Box Test (lag 10): Statistic={lb_stat:.4f}, p-value={lb_p:.4f}")
        print(f"   Autocorrelation Conclusion: {'No significant autocorrelation' if lb_p > 0.05 else 'Significant autocorrelation exists'}")
    except Exception as e:
        print(f"   Cannot perform Ljung-Box test: {e}")
    
    # 3. Heteroscedasticity test
    print("\n3. Heteroscedasticity Test:")
    try:
        # Simple test based on residual variance in first and second half
        n = len(residuals)
        first_half_var = np.var(residuals[:n//2])
        second_half_var = np.var(residuals[n//2:])
        var_ratio = max(first_half_var, second_half_var) / min(first_half_var, second_half_var)
        print(f"   Variance Ratio Test: {var_ratio:.4f}")
        print(f"   Heteroscedasticity Conclusion: {'Homoscedastic' if var_ratio < 2.0 else 'Potential heteroscedasticity'}")
    except Exception as e:
        print(f"   Cannot perform heteroscedasticity test: {e}")
    
    # 4. Durbin-Watson test
    print("\n4. Durbin-Watson Test:")
    dw_stat = simple_durbin_watson(residuals)
    print(f"   DW Statistic: {dw_stat:.4f}")
    if 1.5 < dw_stat < 2.5:
        print("   Conclusion: No significant autocorrelation in residuals")
    elif dw_stat < 1.5:
        print("   Conclusion: Positive autocorrelation exists in residuals")
    else:
        print("   Conclusion: Negative autocorrelation exists in residuals")
    
    # 5. Basic residual statistics
    print("\n5. Basic Residual Statistics:")
    print(f"   Mean: {residuals.mean():.6f}")
    print(f"   Standard Deviation: {residuals.std():.6f}")
    print(f"   Minimum: {residuals.min():.6f}")
    print(f"   Maximum: {residuals.max():.6f}")
    print(f"   Skewness: {stats.skew(residuals):.4f}")
    print(f"   Kurtosis: {stats.kurtosis(residuals):.4f}")
    
    return residuals, standardized_residuals

# Perform residual diagnostics
residuals, std_residuals = comprehensive_residual_analysis(best_model, bitcoin_log, "ARIMA(1,1,0)")

# ================================
# Over-parameterized Model Testing
# ================================

print("\n" + "="*50)
print("OVER-PARAMETERIZED MODEL TESTING")
print("="*50)

# Fit ARIMA(2,1,1) as over-parameterized model
overparameterized_model = ARIMA(bitcoin_log, order=(2, 1, 0)).fit()

print("\nOver-parameterized Model ARIMA(2,1,0) Parameter Estimates:")
print(overparameterized_model.summary().tables[1])

# Check parameter significance
print("\nParameter Significance Test:")
for param_name, param_value in overparameterized_model.params.items():
    p_value = overparameterized_model.pvalues[param_name]
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"{param_name}: {param_value:.4f} (p={p_value:.4f}) {significance}")

residuals, std_residuals = comprehensive_residual_analysis(overparameterized_model, bitcoin_log, "ARIMA(2,1,0)")
