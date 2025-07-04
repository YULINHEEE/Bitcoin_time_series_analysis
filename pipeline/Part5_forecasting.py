# ================================
# Bitcoin Time Series Analysis - Part 5: Forecasting
# ================================

from common_setup import *
print("\n" + "="*50)
print("MODEL FORECASTING")
print("="*50)

# Load best model or fit new model
# with open("pkl/best_model.pkl", "rb") as f:
#     best_model = pickle.load(f)

model = ARIMA(bitcoin_log, order=(1,1,0), trend='n')
best_model = model.fit()

# Set forecast steps
forecast_steps = 12

# Use get_forecast to automatically handle differencing and get confidence intervals
forecast_obj = best_model.get_forecast(steps=forecast_steps)

# Log-level forecasts
forecast_log = forecast_obj.predicted_mean
forecast_ci_log = forecast_obj.conf_int()

# Original-scale forecasts (exponential transformation)
forecast_original = np.exp(forecast_log)
forecast_ci_original = np.exp(forecast_ci_log)

# Generate forecast dates
try:
    last_date = bitcoin_ts.index[-1]
    if isinstance(last_date, tuple):
        last_date = last_date[0]
except:
    last_date = pd.Timestamp('2025-01-01')  # fallback date
forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_steps, freq='M')

# ================================
# Forecast Results Visualization
# ================================

# Log-level forecast plot
plt.figure(figsize=(14,6))
plt.plot(bitcoin_log, label="Historical Log Data")
plt.plot(forecast_dates, forecast_log, label="Forecast (Log)", color='red')
plt.fill_between(forecast_dates,
                 forecast_ci_log.iloc[:,0],
                 forecast_ci_log.iloc[:,1],
                 color='red', alpha=0.3, label="95% CI")
plt.title("Bitcoin Index Forecast Log Level")
plt.xlabel("Time")
plt.ylabel("log(Bitcoin Index)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure 13. Bitcoin Index Forecast Log Level.png", dpi=300, bbox_inches='tight')
plt.show()

# Original-scale forecast plot
plt.figure(figsize=(14,6))
plt.plot(bitcoin_ts, label="Historical Data")
plt.plot(forecast_dates, forecast_original, label="Forecast", color='red')
plt.fill_between(forecast_dates,
                 forecast_ci_original.iloc[:,0],
                 forecast_ci_original.iloc[:,1],
                 color='red', alpha=0.3, label="95% CI")
plt.title("Bitcoin Index Forecast Original Scale (USD)")
plt.xlabel("Time")
plt.ylabel("Bitcoin Index (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure 14. Bitcoin Index Forecast Original Scale (USD).png", dpi=300, bbox_inches='tight')
plt.show()


# ================================
# Forecast Accuracy Assessment
# ================================

print("\n" + "="*50)
print("IN-SAMPLE FORECAST ACCURACY ASSESSMENT")
print("="*50)

# Calculate forecast accuracy metrics
actual = bitcoin_ts.values[1:]
predicted = forecast_original[:len(actual)] if len(forecast_original) > len(actual) else forecast_original
actual = actual[:len(predicted)] if len(actual) > len(predicted) else actual

mae = np.mean(np.abs(actual - predicted))
mse = np.mean((actual - predicted)**2)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actual - predicted) / actual)) * 100
correlation = np.corrcoef(np.array(actual), np.array(predicted))[0, 1]

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Correlation between actual and predicted values: {correlation:.4f}")

# ================================
# Model Summary
# ================================

print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)

print(f"Final selected model: ARIMA(1,1,0)")
print(f"Model AIC: {best_model.aic:.4f}")
print(f"Model BIC: {best_model.bic:.4f}")
print(f"Log-likelihood: {best_model.llf:.4f}")

print("\nModel Parameters:")
for param_name, param_value in best_model.params.items():
    p_value = best_model.pvalues[param_name]
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"  {param_name}: {param_value:.4f} (p={p_value:.4f}) {significance}")

print("\nModel Equation:")
print("log(Bitcoin_t) = log(Bitcoin_{t-1}) + φ₁ * [log(Bitcoin_{t-1}) - log(Bitcoin_{t-2})] + ε_t")
print(f"where φ₁ = {best_model.params.iloc[0]:.4f}")

print("\nAnalysis Complete!")
print("All plots and results have been saved to the current directory.")

# ================================
# Save Results
# ================================

# Build forecast DataFrame
forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "Forecast_Log": forecast_log.values,
    "Log_CI_Lower": forecast_ci_log.iloc[:, 0].values,
    "Log_CI_Upper": forecast_ci_log.iloc[:, 1].values,
    "Forecast_Original": forecast_original.values,
    "Original_CI_Lower": forecast_ci_original.iloc[:, 0].values,
    "Original_CI_Upper": forecast_ci_original.iloc[:, 1].values
})

# Set date as index
forecast_df.set_index("Date", inplace=True)

# Save results
forecast_df.to_csv("data/bitcoin_forecast_results.csv")
with open("pkl/bitcoin_forecast_results.pkl", "wb") as f:
    pickle.dump(forecast_df, f)

print("Forecast results saved to:")
print("  - bitcoin_forecast_results.csv")
print("  - bitcoin_forecast_results.pkl")