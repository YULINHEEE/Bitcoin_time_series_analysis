# Time Series Analysis Part 2: Model Identification

from common_setup import *

# ================================
# Step 1: ACF and PACF Analysis
# ================================
print("\n" + "="*60)
print("STEP 1: ACF AND PACF ANALYSIS")
print("="*60)

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(bitcoin_log_diff, ax=axes[0], lags=40)
axes[0].set_title("ACF of Log-differenced Bitcoin Data")
plot_pacf(bitcoin_log_diff, ax=axes[1], lags=40, method='ywm')
axes[1].set_title("PACF of Log-differenced Bitcoin Data")
plt.tight_layout()
#plt.savefig('Figure 9. ACF_PACF_identification.png', dpi=300, bbox_inches='tight')
#plt.show()

# Define candidate models based on ACF/PACF patterns
acf_pacf_candidates = [
    (0, 1, 1), (0, 1, 2), (1, 1, 0), (1, 1, 1), 
    (1, 1, 2), (2, 1, 1), (2, 1, 2)
]

print("\nCandidate models from ACF/PACF analysis:")
for i, order in enumerate(acf_pacf_candidates, 1):
    print(f"  {i:2d}. ARIMA{order}")

# ================================
# Step 2: BIC Matrix (EACF Alternative)
# ================================
print("\n" + "="*60)
print("STEP 2: BIC MATRIX ANALYSIS (EACF ALTERNATIVE)")
print("="*60)

# Build BIC matrix for model comparison
max_p, max_q = 5, 5
bic_matrix = pd.DataFrame(index=range(max_p + 1), columns=range(max_q + 1))

print("Calculating BIC values for ARIMA(p,0,q) models...")
for p in range(max_p + 1):
    for q in range(max_q + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(bitcoin_log_diff, order=(p, 0, q)).fit()
            bic_matrix.iloc[p, q] = model.bic
            print(f"ARIMA({p},0,{q}): BIC = {model.bic:.2f}")
        except:
            bic_matrix.iloc[p, q] = np.nan
            print(f"ARIMA({p},0,{q}): Failed")

bic_matrix = bic_matrix.astype(float)

# Visualize BIC matrix
plt.figure(figsize=(8, 6))
sns.heatmap(bic_matrix, annot=True, fmt=".1f", cmap="YlGnBu", 
            cbar_kws={"label": "BIC"})
plt.title("BIC Values for ARIMA(p,1,q) Models")
plt.xlabel("q (MA order)")
plt.ylabel("p (AR order)")
#plt.savefig("Figure 10. BIC_heatmap.png", dpi=300, bbox_inches='tight')
#plt.show()

# ================================
# Step 3: Top BIC Models Selection
# ================================
print("\n" + "="*60)
print("STEP 3: TOP BIC MODELS SELECTION")
print("="*60)

# Extract top models by BIC
bic_candidates = []
for p in range(max_p + 1):
    for q in range(max_q + 1):
        bic_val = bic_matrix.iloc[p, q]
        try:
            # Convert to float using string conversion to avoid type issues
            bic_float = float(str(bic_val))
            if not pd.isna(bic_val) and np.isfinite(bic_float):
                bic_candidates.append(((p, 1, q), bic_float))
        except:
            continue

# Sort by BIC (ascending = better)
bic_candidates = sorted(bic_candidates, key=lambda x: x[1])
top_bic_models = bic_candidates[:10]

print("Top 10 ARIMA(p,1,q) models by BIC:")
for i, (order, bic_val) in enumerate(top_bic_models, 1):
    print(f"  {i:2d}. ARIMA{order} - BIC: {bic_val:.2f}")

top_bic_orders = [order for order, _ in top_bic_models]

# ================================
# Save Results for Model Fitting
# ================================
print("\n" + "="*60)
print("SAVING IDENTIFICATION RESULTS")
print("="*60)

# Combine all candidate models
final_candidates = list(set(acf_pacf_candidates + top_bic_orders))

# Save to CSV
candidate_df = pd.DataFrame(final_candidates, columns=["p", "d", "q"])
candidate_df["Model"] = candidate_df.apply(lambda x: f"ARIMA({x['p']},{x['d']},{x['q']})", axis=1)
#candidate_df.to_csv("candidate_models.csv", index=False)

# Save to pickle for next script
results_dict = {
    "bic_matrix": bic_matrix.to_dict(),
    "final_candidates": final_candidates,
    "acf_pacf_candidates": acf_pacf_candidates,
    "top_bic_models": top_bic_models,
    "bitcoin_log_diff": bitcoin_log_diff
}

SAVE_RESULTS = False

if SAVE_RESULTS:
    with open("model_identification_results.pkl", "wb") as f:
        pickle.dump(results_dict, f)
    print("Results saved.")

print(f"{len(final_candidates)} candidate models saved to:")
print("  - candidate_models.csv")
print("  - model_identification_results.pkl")
