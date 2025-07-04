# Time Series Analysis Part 3: Model Fitting and Comparison

from common_setup import *
# Part3_model_fitting.py
import pickle

with open("pkl/model_identification_results.pkl", "rb") as f:
    identification_results = pickle.load(f)

final_candidates = identification_results["final_candidates"]

# ================================
# Model Fitting Function
# ================================

def fit_arima_models(data, orders):
    """Fit multiple ARIMA models and return results"""
    results = {}
    
    print(f"\n{'='*60}")
    print("MODEL FITTING AND ESTIMATION")
    print(f"{'='*60}")
    
    for order in orders:
        model_name = f"ARIMA{order}"
            
        try:
            print(f"\nFitting {model_name}...")
            model = ARIMA(data, order=order).fit()
                
            results[model_name] = {
                'model': model,
                'order': order,
                'aic': model.aic,
                'bic': model.bic,
                'loglik': model.llf,
                'params': model.params,
                'pvalues': model.pvalues,
                'n_obs': model.nobs,
                'fitted_values': model.fittedvalues,
                'residuals': model.resid
            }
                
            # Print fitting results
            print(f"{model_name} fitted successfully")
            # Print parameter estimates with significance
            print("  Parameter estimates:")
            for param_name, param_value in model.params.items():
                p_value = model.pvalues[param_name]
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"    {param_name}: {param_value:.4f} (p={p_value:.4f}) {significance}")
                
        except Exception as e:
            print(f"Error fitting {model_name}: {str(e)}")
            continue
    
    print(f"\n Model fitting completed. {len(results)} models fitted successfully.")
    return results

# ================================
# Fit All Candidate Models
# ================================

# Fit models using ML methods
model_results = fit_arima_models(bitcoin_log, final_candidates)

# ================================
# Model Comparison and Selection
# ================================

def compare_models(results, criterion='aic' or 'bic'):
    """Compare models and sort by specified criterion"""
    model_comparison = []
    
    for model_name, result in results.items():
        
        if criterion.lower() == 'aic':
            score = result['aic']
        elif criterion.lower() == 'bic':
            score = result['bic']
        else:
            score = result['aic']
        
        model_comparison.append({
            'Model': model_name,
            'Order': result['order'],
            'AIC': result['aic'],
            'BIC': result['bic'],
            'LogLik': result['loglik'],
            'N_Obs': result['n_obs'],
            'Score': score
        })   
    # Convert to DataFrame and sort
    df_comparison = pd.DataFrame(model_comparison)
    df_comparison = df_comparison.sort_values('Score').reset_index(drop=True)
    
    return df_comparison

# ================================
# Model Comparison Results
# ================================

print(f"\n{'='*60}")
print("MODEL COMPARISON RESULTS")
print(f"{'='*60}")

# AIC comparison
print("\nAIC Model Ranking:")
print("-" * 80)
aic_comparison = compare_models(model_results, 'aic')
print(aic_comparison[['Model', 'Order', 'AIC', 'BIC', 'LogLik']].round(4))

# ================================
# Visualization of Model Comparison
# ================================

# Create comparison plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# AIC comparison plot
models_aic = aic_comparison['Model'].values
aic_values = aic_comparison['AIC'].values

bars1 = ax1.bar(range(len(models_aic)), aic_values, alpha=0.7, color='steelblue')
ax1.set_xlabel('Model')
ax1.set_ylabel('AIC Value')
ax1.set_title('Model Comparison by AIC')
ax1.set_xticks(range(len(models_aic)))
ax1.set_xticklabels(models_aic, rotation=45, ha='right')
ax1.grid(True, alpha=0.3)

# Highlight best model
bars1[0].set_color('red')
bars1[0].set_alpha(0.9)

# BIC comparison plot
bic_comparison = compare_models(model_results, 'bic')
models_bic = bic_comparison['Model'].values
bic_values = bic_comparison['BIC'].values

bars2 = ax2.bar(range(len(models_bic)), bic_values, alpha=0.7, color='steelblue')
ax2.set_xlabel('Model')
ax2.set_ylabel('BIC Value')
ax2.set_title('Model Comparison by BIC')
ax2.set_xticks(range(len(models_bic)))
ax2.set_xticklabels(models_bic, rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

# Highlight best model (lowest BIC)
bars2[0].set_color('red')
bars2[0].set_alpha(0.9)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='steelblue', alpha=0.7, label='Candidate Models'),
    Patch(facecolor='red', alpha=0.9, label='Best Model')
]
ax1.legend(handles=legend_elements, loc='upper left')
ax2.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
#plt.savefig('Figure 11. Model_comparison_fitting.png', dpi=300, bbox_inches='tight')
#plt.show()

# ================================
# Best Model Selection
# ================================

print(f"\n{'='*60}")
print("BEST MODEL SELECTION")
print(f"{'='*60}")

best_model_aic = aic_comparison.iloc[0]
best_model_bic = bic_comparison.iloc[0]

print(f"Best model by AIC: {best_model_aic['Model']} - AIC: {best_model_aic['AIC']:.4f}")
print(f"Best model by BIC: {best_model_bic['Model']} - BIC: {best_model_bic['BIC']:.4f}")

# Select best model based on BIC (more parsimonious)
best_model_name = best_model_bic['Model']
best_model = model_results[best_model_name]['model']
best_order = model_results[best_model_name]['order']

print(f"\n Selected final model: {best_model_name}")
print(f"  Order: ARIMA{best_order}")
print(f"  AIC: {best_model.aic:.4f}")
print(f"  BIC: {best_model.bic:.4f}")

print(f"\n{'='*40}")
print("DETAILED MODEL SUMMARY")
print(f"{'='*40}")
print(best_model.summary())

# ================================
# Accuracy Assessment
# ================================

def calculate_accuracy_metrics(model, original_data):
    """Calculate comprehensive accuracy metrics"""
    fitted_values = model.fittedvalues
    residuals = model.resid
    
    # Basic accuracy metrics
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE carefully
    try:
        # Use original data length that matches fitted values
        actual_values = original_data.iloc[-len(fitted_values):]
        mape = np.mean(np.abs((actual_values - fitted_values) / actual_values) * 100)
    except:
        mape = np.nan
    
    # Additional metrics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'Mean_Residual': mean_residual,
        'Std_Residual': std_residual,
        'N_Obs': len(residuals)
    }

print(f"\n{'='*60}")
print("ACCURACY ASSESSMENT FOR ALL MODELS")
print(f"{'='*60}")

accuracy_results = []
for model_name, result in model_results.items():
    try:
        accuracy = calculate_accuracy_metrics(result['model'], bitcoin_log)
        accuracy_results.append({
            'Model': model_name,
            'Order': str(result['order']),
            'MAE': accuracy['MAE'],
            'MSE': accuracy['MSE'],
            'RMSE': accuracy['RMSE'],
            'MAPE': accuracy['MAPE'],
            'Mean_Residual': accuracy['Mean_Residual'],
            'Std_Residual': accuracy['Std_Residual']
        })
    except Exception as e:
        print(f"Error calculating accuracy for {model_name}: {e}")

accuracy_df = pd.DataFrame(accuracy_results)
accuracy_df = accuracy_df.sort_values('RMSE').reset_index(drop=True)

print("\nAccuracy Metrics for All Models (sorted by RMSE):")
print("-" * 100)
print(accuracy_df.round(6))

# Best model accuracy
best_accuracy = calculate_accuracy_metrics(best_model, bitcoin_log)
print(f"\n{'='*40}")
print(f"BEST MODEL ACCURACY ({best_model_name})")
print(f"{'='*40}")
for metric, value in best_accuracy.items():
    if metric != 'N_Obs':
        print(f"{metric:15s}: {value:.6f}")
    else:
        print(f"{metric:15s}: {value}")

# ================================
# Save Fitting Results
# ================================

print(f"\n{'='*60}")
print("SAVING FITTING RESULTS")
print(f"{'='*60}")

# Save comprehensive results
fitting_results = {
    'best_model_name': best_model_name,
    'best_model_order': best_order,
    'best_model_summary': str(best_model.summary()),
    'aic_comparison': aic_comparison.to_dict('records'),
    'bic_comparison': bic_comparison.to_dict('records'),
    'accuracy_results': accuracy_df.to_dict('records'),
    'best_accuracy': best_accuracy,
    'model_count': len(model_results)
}


# Save comparison tables as CSV and best model
SAVE_RESULTS = True

if SAVE_RESULTS:
    with open('data/aic_model_comparison.csv', 'w') as f:
        aic_comparison.to_csv(f, index=False)
    with open('data/bic_model_comparison.csv', 'w') as f:
        bic_comparison.to_csv(f, index=False)
    with open('data/model_accuracy_comparison.csv', 'w') as f:
        accuracy_df.to_csv(f, index=False)
    with open('pkl/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    with open('pkl/model_fitting_results.pkl', 'wb') as f:
        pickle.dump(fitting_results, f)

print("Fitting results saved to:")
print("  - model_fitting_results.pkl")
print("  - aic_model_comparison.csv")
print("  - bic_model_comparison.csv")
print("  - model_accuracy_comparison.csv")
print("  - best_model.pkl")

print(f"\n{'='*60}")
print("MODEL FITTING AND COMPARISON COMPLETED!")
print(f"{'='*60}")
print(f"Total models fitted: {len(model_results)}")
print(f"Best model: {best_model_name}")
print(f"Final selected order: ARIMA{best_order}")
