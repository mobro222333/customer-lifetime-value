# Customer Lifetime Value (CLV) Prediction

Predicts how much revenue a customer will generate in the future, using a
two-stage model built on the Online Retail II dataset.

## Overview

Rather than predicting a single CLV number directly, this project splits the
problem into two stages to handle the fact that most customers do not
purchase again in any given future window (zero-inflated target):

1. **Stage 1 — Classifier:** predicts the probability that a customer makes
   at least one purchase in the future window, using RFM-based features.
2. **Stage 2 — Regressor:** predicts how much a customer will spend,
   *conditional on making a purchase* (trained only on customers who did).

The final **expected CLV** for a customer is:

```
expected_clv = purchase_probability × predicted_spend_if_buying
```

## Project Structure

```
.
├── notebook.ipynb              # Full pipeline: cleaning → RFM → modeling → SHAP → segmentation
├── app.py                      # Streamlit app for interactive/bulk scoring
├── requirements.txt            # Python dependencies
├── stage1_classifier.pkl       # Trained purchase-probability classifier
├── stage2_regressor.pkl        # Trained conditional-spend regressor
├── feature_cols.json           # Exact feature order expected by both models
├── clv_tier_thresholds.json    # Quartile cutoffs used to bucket customers into CLV tiers
├── tier_summary.csv            # Average RFM profile per CLV tier
└── customer_results.csv        # Scored customer table from the training run
```

## Pipeline Summary (Notebook)

1. **Data cleaning** — load Online Retail II, drop missing `CustomerID`,
   remove cancellations and non-positive quantities/prices, compute revenue.
2. **Time-based split** — features are built only from data *before* a
   cutoff date; the target is built only from data *after* it, to avoid
   leakage.
3. **RFM feature engineering** — recency, frequency, monetary value, tenure,
   basket size, product diversity, and derived ratios
   (`spend_per_tenure_day`, `purchase_rate_per_month`).
4. **Cohort analysis** — retention heatmap by acquisition month, used to
   sanity-check the churn pattern found in the target distribution.
5. **Two-stage modeling** — `GradientBoostingClassifier` +
   `GradientBoostingRegressor`, both tuned with `GridSearchCV`.
6. **Explainability** — SHAP summary plots for both stages.
7. **Segmentation** — customers bucketed into Low/Medium/High/Top CLV tiers
   by quartile, with RFM profiles compared across tiers.
8. **Evaluation** — ROC-AUC (classifier), MAE/R² in both log and dollar
   scale (regressor), predicted-vs-actual and ROC visualizations.

## Running the App

The app is a flat deployment — all files sit in one folder, no subfolders.

```bash
pip install -r requirements.txt
streamlit run app.py
```

**App features:**
- **Score a Customer** — manually enter RFM values for a live prediction.
- **Bulk Upload** — upload a CSV of customers to score all at once, with a
  downloadable results file.
- **Explore Segments** — browse tier profiles and existing scored customers.

## Key Design Decisions

- **Time-based split, not random split** — simulates a real forecasting
  scenario and prevents the model from seeing future purchases as features.
- **Two-stage model over a single regressor** — the target is zero-inflated
  (a majority of customers have $0 future spend), so a single regression
  model trained on `log1p(spend)` would still struggle to distinguish
  churners from low spenders. Separating "will they buy" from "how much"
  lets each model specialize.
- **Log-transformed regression target** — future spend is heavily
  right-skewed; training on `log1p(spend)` and inverting with `expm1()`
  at prediction time prevents a small number of high-spending outliers
  from dominating the loss function.
- **Saved feature order and tier thresholds** — ensures any new customer
  scored later (e.g. through the app) is processed identically to how the
  training data was processed.

## Business Impact

- **Top-tier customers** (top 25% by expected CLV) are strong candidates
  for loyalty programs or early access to new products.
- **At-risk high-value customers** — high past spend but low predicted
  purchase probability — are flagged as win-back campaign candidates,
  since they represent proven value at risk of churning.
- Model performance is reported in business terms (ROC-AUC for ranking
  likely buyers, dollar-scale MAE for spend estimates) rather than only
  raw ML metrics.
