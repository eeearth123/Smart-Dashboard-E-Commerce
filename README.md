# 💰 Churn Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Same campaign. Same cost per customer. 78x better ROI.**

But here's the key: we didn't just predict churn — we simulated what would happen if we intervened.

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **Dataset Size** | 58,000+ orders (Olist E-Commerce) |
| **Model** | LightGBM (AutoML via FLAML) |
| **Macro F1** | 0.658 |
| **Churn Recall** | 78% |
| **ROI Improvement** | +3.4% → +266.6% (78x) |
| **Budget Saved** | 97% (R$895K → R$25K) |

---

## 📉 The Problem

We built a churn prediction model (LightGBM, 58K orders).  
Initial idea: offer free shipping to everyone.  
**But would it work? And for whom?**

---

## 🧠 The Approach

Instead of just predicting churn, we built a **simulation engine** that answers: *"What if we intervene?"*

1. **Take customer's feature vector**
2. **Modify feature** (e.g., set `freight_value = 0` for free shipping)
3. **Re-run `predict_proba()`** to get new churn probability
4. **Calculate ROI**: `[(Value of saved customers - Budget) / Budget] × 100`

This isn't causal inference — it's a **directional estimate** that lets us identify who is most likely to respond to a specific intervention.

---

## 🎯 Strategic Segmentation

We combined:
- ✅ **AI confidence** (probability thresholds)
- ✅ **Business context** (category profitability)
- ✅ **Customer behavior patterns**

**Different segments → Different strategies**

---

## 🔢 Results

### 🟡 Untargeted Approach (Send to Everyone)

| Metric | Value |
|--------|-------|
| **Customers** | 45,791 |
| **Budget** | 45,791 × R$34 = **~R$895,118** |
| **Success Rate** | 24.7% (6,491 customers flipped from Churn → Stay) |
| **Break-even** | 23.6% (R$34 cost / R$144 avg. order value) |
| **ROI** | **+3.4%** (barely above break-even) |
| **Result** | Profitable, but razor-thin margin + massive capital exposure |

### 🟢 ML-Targeted Approach (High-Risk + Profitable Categories)

| Metric | Value |
|--------|-------|
| **Customers** | 5,363 (filtered by prob > 0.75 + category profitability) |
| **Budget** | 5,363 × R$34 = **R$25,112** (97% less!) |
| **Success Rate** | 46.4% (model + targeting = higher response) |
| **Break-even** | 13.5% (higher avg. order value segment = easier to profit) |
| **ROI** | **+266.6%** (much higher margin above break-even) |
| **Result** | Same campaign, dramatically better outcome |

---

## 💡 The Takeaway

> **The model doesn't just predict who will churn — it identifies who will actually respond to a specific intervention.**

Targeting the right segment with the right strategy matters more than the campaign itself.

---

## ⚠️ Caveat

These numbers depend entirely on:
- ✅ Model quality (ours: Macro F1 = 0.658 on imbalanced data)
- ✅ Feature engineering that reflects real-world levers
- ✅ Reasonable assumptions in the simulation

**If your model is weak → this approach amplifies mistakes.**  
**If your model is strong → this approach multiplies impact.**

---

## 📊 Final Thoughts

| Metric | Value |
|--------|-------|
| **Accuracy** | 74.74% |
| **Macro F1** | 0.6742 |

Metrics alone don't drive action. If we can't explain **WHY**, business teams won't trust it.

That's why we used:
- ✅ **SHAP values** for interpretability
- ✅ **Feature importance analysis**
- ✅ **Probability distribution visualization**
- ✅ **Campaign simulation with clear ROI**

> **A model is only as good as your ability to explain it.**


