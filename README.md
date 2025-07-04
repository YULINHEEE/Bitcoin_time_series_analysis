# Bitcoin Time Series Forecasting Project

This project performs time series modeling and forecasting for the Bitcoin Index using monthly data. The analysis follows a classical ARIMA modeling pipeline, including stationarity testing, model identification, model fitting, residual diagnostics, and multi-step forecasting.

---

## Project Structure
```
BITCOINTIMESERIESANALYSIS/
│
├── data/ # Input and comparison data
│ ├── bic_model_comparison.csv
│ ├── aic_model_comparison.csv
│ ├── bitcoin_12month_forecasts.csv
│ ├── bitcoin_forecast_results.csv
│ ├── bitcon_index_data2025.csv
│ └── model_accuracy_comparison.csv
│
├── pipeline/ # Core Python pipeline scripts
│ ├── common_setup.py # Global setup, plotting style, fonts
│ ├── install_requirements.py # pip installer for required packages
│ ├── Part1_stationarity_test_data_transformation.py
│ ├── Part2_model_identification.py
│ ├── Part3_model_fitting.py
│ ├── Part4_residual_diagnostics.py
│ └── Part5_forecasting.py
│
├── pkl/ # Saved intermediate artifacts (via pickle)
│ ├── best_model.pkl
│ ├── bitcoin_forecast_results.pkl
│ ├── bitcoin_forecast_summary.pkl
│ ├── bitcoin_log_diff.pkl
│ ├── bitcoin_log.pkl
│ ├── model_fitting_results.pkl
│ ├── model_identification_results.pkl
│ └── residual_diagnostics_results.pkl
│
├── results_pics/ # Saved figures (optional output folder)
│ └── (you may save plots here if enabled)
│
├── R_code/ # Additional R scripts
│ ├── bitcoin_analysis.R
│
├── README.md
└── requirements.txt # Python environment dependencies
```

## Project Objective

The goal is to model and forecast monthly Bitcoin Index values using ARIMA methodology. The dataset exhibits strong non-stationarity and exponential growth, hence a log transformation and differencing were applied.

The best identified model is: ARIMA(1,1,0).

## How to Run

### Step-by-step execution:

1. **Environment Setup**

pip install -r requirements.txt


2. **Run Analysis Pipeline**

Please follow the order:
python pipeline/Part1_stationarity_test_data_transformation.py
python pipeline/Part2_model_identification.py
python pipeline/Part3_model_fitting.py
python pipeline/Part4_residual_diagnostics.py
python pipeline/Part5_forecasting.py


3. **Supplementary R Code Provided**
In addition to the full Python forecasting pipeline, we also provide a partial implementation in R.

R_code/bitcoin_analysis.R includes:

- Data loading and exploration

- Stationarity testing

- ADF tests and differencing

- Partial ACF/PACF plotting, EACF, BIC table

- Model order suggestion and diagnostics

- Note: Forecasting is only implemented in Python. This R script is intended for cross-validation, demonstration, and comparison purposes.
