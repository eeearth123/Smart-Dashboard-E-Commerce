# 🛒 Olist Customer Churn Prediction — From Model to Production Dashboard

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-orange)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)]()
[![BigQuery](https://img.shields.io/badge/Data-BigQuery-4285F4)]()

## Overview

End-to-end churn prediction pipeline built on the Brazilian e-commerce 
dataset (Olist). The model is trained offline with AutoML, then served 
live through a Streamlit dashboard connected to BigQuery.

**Macro F1: 0.6784 | Threshold: 0.55 | Algorithm: LightGBM**

---

## Architecture: Model → Dashboard

```
BigQuery (45,786 rows)
       │
       ▼
process_features()          ← Feature engineering at load time
  ├─ delivery_days          (order_delivered - order_purchased)
  ├─ delivery_vs_estimated  (estimated - actual delivery)
  ├─ freight_ratio          (freight_value / price)
  ├─ avg_purchase_gap       (median days between repeat orders)
  ├─ gap_real / gap_vs_avg  (repeat-buyer behavior signals)
  ├─ first_purchase_late    (first impression signal)
  ├─ first_purchase_bad_review
  └─ is_extremely_late
       │
       ▼
lgbm_model.predict_proba()  ← 15 features, threshold = 0.55
       │
       ▼
churn_probability + status  ← Rule-based segmentation overlay
  ├─ Lost        (lateness_score > 3.0)
  ├─ High Risk   (AI prob > 0.75)
  ├─ Warning     (lateness > 1.5)
  ├─ Medium Risk (AI prob 0.40–0.75)
  └─ Active
       │
       ▼
Streamlit Dashboard (7 pages)
  ├─ Business Overview    — Revenue trend, top categories
  ├─ Churn Overview       — Rule vs AI comparison trend
  ├─ Action Plan          — Model-driven campaign simulator
  ├─ Buying Cycle         — Repurchase gap analysis
  ├─ Logistics Insights   — State-level delivery map (pydeck)
  ├─ Seller Audit         — Per-seller churn attribution
  └─ Customer Detail      — Individual drill-down
```

---

## Key Feature: Model-Driven Campaign Simulator (Page 3)

The Action Plan page simulates campaign ROI by **modifying feature 
values and re-running predict_proba()** — not by using a fixed lift 
assumption.

```python
# Example: "What if we offer free shipping?"
df_sim['freight_value'] = 0      # simulate the campaign
df_sim['freight_ratio'] = 0
prob_sim = model.predict_proba(X_sim)[:, 1]

uplift = prob_orig - prob_sim    # positive = risk reduced
success_rate = (uplift > 0.08).mean()
```

This gives a data-grounded success rate rather than a manual estimate,
making ROI calculation directly tied to the model's learned behavior.

---

## Feature Importance (Top 5)

| Rank | Feature               | Importance |
|------|-----------------------|------------|
| 1    | freight_value         | 19.6%      |
| 2    | price                 | 18.8%      |
| 3    | freight_ratio         | 11.7%      |
| 4    | delivery_vs_estimated | 9.6%       |
| 5    | payment_installments  | 8.6%       |

→ ~50% of churn is driven by **price + freight**
→ ~18% by **delivery timing**
→ Informs which campaigns have highest expected uplift

---

## Model Performance

| Threshold | Macro F1 | Churn Recall | Stay Recall |
|-----------|----------|--------------|-------------|
| 0.50      | 0.6742   | 82.5%        | 56.5%       |
| **0.55**  | **0.6583** | **78.1%**  | **60.9%**   |
| 0.60      | 0.6093   | 67.8%        | 67.2%       |

Threshold 0.55 selected: balances catching churners (78%) 
while limiting false positives on retain-spend.

---

## Stack

- **Model**: LightGBM via AutoML (flaml), 5-fold CV, macro F1
- **Data**: Google BigQuery (`asia-southeast1`)
- **Dashboard**: Streamlit Cloud
- **Viz**: Altair, Pydeck
- **Serving**: `model.predict_proba()` called on every page load

## Files

```
app.py                              ← Streamlit dashboard (7 pages)
olist_churn_model_final (1).pkl     ← Trained LightGBM model
model_features_final (1).pkl        ← Feature name list (15 features)
```
