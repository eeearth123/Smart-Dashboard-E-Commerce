# 🤖 Model Report — Olist Churn Prediction

> **Algorithm:** LightGBM (selected by FLAML AutoML)  
> **Macro F1:** 0.6784 | **Threshold:** 0.55 | **Features:** 15

---

## 📋 สารบัญ

| ส่วน | เนื้อหา |
|------|---------|
| [1](#1--model-overview) | Model Overview |
| [2](#2--training-setup) | Training Setup |
| [3](#3--feature-set) | Feature Set (15 ตัว) |
| [4](#4--performance-metrics) | Performance Metrics |
| [5](#5--shap-analysis) | SHAP Analysis |
| [6](#6--segment-profile) | Segment Profile |
| [7](#7--counter-intuitive-findings) | Counter-intuitive Findings |
| [8](#8--limitations) | Limitations |

---

## 1 · Model Overview

```
Dataset  : Olist Brazilian E-Commerce (~58K orders)
Task     : Binary Classification — Churn (1) vs Stay (0)
Churn Definition : ไม่ซื้อซ้ำภายใน 180 วัน
                   (data-driven: ครอบคลุม ~77% ของ repeat buyers)
Algorithm: LightGBM  ←  FLAML AutoML เลือกให้อัตโนมัติ
Metric   : Macro F1  ←  เหมาะกับ class imbalance
```

**Class distribution:**

| Class | จำนวน | สัดส่วน |
|-------|--------|---------|
| Churn (1) | ~77,858 | **84%** |
| Stay (0)  | ~40,444 | **16%** |

→ Severe imbalance แก้ด้วย **Sample Weights** (Churn = 0.626, Stay = 2.482)

---

## 2 · Training Setup

| Parameter | Value |
|-----------|-------|
| Library | FLAML AutoML |
| Best Estimator | LightGBM (`lgbm`) |
| CV Strategy | 5-fold cross-validation |
| Optimize Metric | macro_f1 |
| Budget | 900 seconds |
| NaN Handling | Fill median (พบ 3 ค่า) |
| Class Imbalance | Sample weighting |
| Threshold Tuning | แยกหลัง training |

---

## 3 · Feature Set

**15 features แบ่งเป็น 5 กลุ่ม:**

### 3.1 Logistics Features

| Feature | สูตร | ความหมาย |
|---------|------|----------|
| `delivery_days` | delivered − purchased | ใช้เวลาส่งจริงกี่วัน |
| `estimated_days` | estimated − purchased | กี่วันตามที่บอกไว้ |
| `delivery_vs_estimated` | estimated − actual | บวก = เร็วกว่ากำหนด |
| `delay_days` | delivered − estimated | บวก = ช้ากว่ากำหนด |

### 3.2 Price & Freight Features

| Feature | สูตร | ความหมาย |
|---------|------|----------|
| `price` | ราคาสินค้า | — |
| `freight_value` | ค่าส่งจริง (R$) | — |
| `freight_ratio` | freight / price | สัดส่วนค่าส่งต่อราคา |
| `payment_installments` | จำนวนงวดผ่อน | — |

### 3.3 Purchase Behaviour Features

| Feature | ความหมาย |
|---------|----------|
| `is_first_purchase` | 1 = ออเดอร์แรกของลูกค้าคนนี้ |
| `is_repeat_buyer` | 1 = ซื้อมากกว่า 1 ครั้ง |
| `avg_purchase_gap` | ค่าเฉลี่ยวันระหว่างออเดอร์ |
| `gap_real` | gap จริง (0 สำหรับ first-time) |
| `gap_vs_avg_real` | gap − avg_gap (0 สำหรับ first-time) |

### 3.4 First Impression Features

| Feature | ความหมาย |
|---------|----------|
| `first_purchase_late` | 1 = ออเดอร์แรกส่งช้ากว่ากำหนด |
| `first_purchase_bad_review` | 1 = ออเดอร์แรกได้รีวิว ≤ 2 ดาว |
| `is_extremely_late` | 1 = ช้ากว่ากำหนดมากกว่า 7 วัน |

### 3.5 Review Quality Features

| Feature | ความหมาย |
|---------|----------|
| `is_low_score` | 1 = review score ≤ 2 |
| `is_high_score` | 1 = review score = 5 |

---

## 4 · Performance Metrics

### Threshold Tuning Results

| Threshold | Macro F1 | Churn Recall | Stay Recall | Accuracy |
|-----------|----------|--------------|-------------|----------|
| 0.30 | 0.6550 | 95.8% | 30.2% | 83.0% |
| 0.35 | 0.6777 | 92.4% | 39.5% | 82.2% |
| 0.40 | 0.6776 | 89.1% | 45.2% | 80.6% |
| 0.45 | **0.6804** | 85.9% | 51.7% | 79.2% |
| 0.50 | 0.6742 | 82.5% | 56.5% | 77.5% |
| **0.55** | 0.6583 | **78.1%** | **60.9%** | 74.8% |
| 0.60 | 0.6093 | 67.8% | 67.2% | 67.7% |
| 0.65 | 0.5216 | 49.9% | 77.3% | 55.2% |

> **เลือก Threshold = 0.55** เพราะ:
> - Churn Recall 78% — จับลูกค้าเสี่ยงได้มากพอ
> - Stay Recall 61% — ไม่เสียงบกับคนที่ไม่ได้ churn มากเกินไป
> - Cost of False Negative (พลาด churner) > Cost of False Positive (ยิงคนไม่ churn)

### Confusion Matrix (threshold = 0.55, test set n = 14,503)

```
                  Pred Stay    Pred Churn
Actual Stay        1,590  ✅    1,224  ❌  (FP 43%)
Actual Churn       2,044  ❌    9,645  ✅  (Recall 78%)
```

### Final Scores

| Metric | Value |
|--------|-------|
| Macro F1 | **0.6583** |
| Churn Precision | 0.887 |
| Churn Recall | 0.781 |
| Stay Precision | 0.437 |
| Stay Recall | 0.609 |
| AUC-ROC | **~0.85** (จาก SHAP analysis) |

---

## 5 · SHAP Analysis

> `![SHAP Beeswarm](images/shap_beeswarm.png)`

### 5.1 Feature Importance เทียบกับ SHAP

**ทำไม Feature Importance กับ SHAP ไม่ตรงกัน?**

| วิธีวัด | วัดอะไร | ผล |
|---------|---------|-----|
| **Feature Importance (split-based)** | จำนวนครั้งที่ feature ถูกใช้แบ่ง node | `freight_value` #1 (19.6%) |
| **SHAP (mean \|SHAP\|)** | impact จริงต่อแต่ละ prediction | `is_first_purchase` #1 |

→ `is_first_purchase` เป็น binary (0/1) ใช้แบ่งน้อยครั้ง แต่เมื่อ = 1 มี impact สูงมาก

### 5.2 Global Feature Importance (Mean |SHAP|)

| Rank | Feature | Mean \|SHAP\| | SHAP Churn | SHAP Stay |
|------|---------|---------------|------------|-----------|
| 1 | `is_first_purchase` | ~0.253 | +0.135 | −0.315 |
| 2 | `avg_purchase_gap` | ~0.251 | +0.163 | −0.220 |
| 3 | `price` | ~0.248 | +0.096 | −0.143 |
| 4 | `payment_installments` | ~0.246 | +0.066 | −0.081 |
| 5 | `freight_value` | ~0.196 | +0.032 | −0.027 |

> `![SHAP Bar](images/shap_bar.png)`

### 5.3 Churn vs Stay — Mean SHAP per Group

> `![SHAP Group Compare](images/shap_group_compare.png)`

**กลุ่ม Churn ถูกดันโดย:**
- `avg_purchase_gap` SHAP +0.163 — gap ยิ่งนาน ยิ่งเสี่ยง
- `is_first_purchase` SHAP +0.135 — first-time buyer เสี่ยงสูงมาก
- `price` SHAP +0.096 — สินค้าแพงมักซื้อครั้งเดียว

**กลุ่ม Stay ถูกปกป้องโดย:**
- `is_first_purchase` SHAP −0.315 — repeat buyer ลด risk มาก
- `avg_purchase_gap` SHAP −0.220 — ซื้อสม่ำเสมอ = loyal
- `price` SHAP −0.143 — ราคาปานกลาง repeat ได้ง่ายกว่า

### 5.4 Dependence Plots

> `![SHAP Dependence](images/shap_dependence.png)`

**อ่านได้ว่า:**
- `price` — linear positive: ยิ่งแพง SHAP ยิ่งบวก (เพิ่ม churn risk)
- `delivery_vs_estimated` — ยิ่งส่งช้ากว่ากำหนด SHAP ยิ่งบวก
- `avg_purchase_gap` — ยิ่งนาน SHAP ยิ่งบวก (แต่มี threshold ประมาณ 20+ วัน)

### 5.5 Waterfall — รายคน

> `![SHAP Waterfall](images/shap_waterfall.png)`

| กลุ่ม | churn_prob | ตัวขับเคลื่อนหลัก |
|-------|-----------|-----------------|
| High Risk (>0.9) | 0.914 | price สูง + freight_value + is_first |
| Borderline Churn (0.55–0.65) | 0.567 | is_low_score + payment_installments |
| Borderline Stay (0.45–0.55) | 0.473 | freight_ratio ลด risk บางส่วน |
| Safe Stay (<0.2) | 0.185 | is_first=0 + avg_purchase_gap ต่ำ |

---

## 6 · Segment Profile

> `![Segment Heatmap](images/segment_heatmap.png)`

| Feature | Safe < 0.4 | Medium 0.4–0.55 | High 0.55–0.75 | Very High > 0.75 |
|---------|-----------|----------------|---------------|-----------------|
| price (R$) | **77** | 89 | 100 | **262** |
| is_first_purchase | **33%** | 73% | 98% | **95%** |
| avg_purchase_gap (วัน) | **4.1** | 0.85 | 0.53 | **15.3** |
| freight_ratio | 0.41 | 0.42 | 0.32 | **0.15** |
| delivery_days | 10.7 | 12.0 | 12.0 | **13.9** |
| is_low_score | **33%** | 32% | 7% | 8% |
| is_high_score | 43% | 46% | 59% | **63%** |

**อ่านได้ว่า:**
- กลุ่ม Very High Risk = ซื้อสินค้าแพง (R$262), first-time buyer 95%, ส่งช้ากว่า
- กลุ่ม Safe = repeat buyer, ราคาปานกลาง (R$77), ซื้อบ่อย (gap 4 วัน)

---

## 7 · Counter-intuitive Findings

### 7.1 `freight_ratio` ต่ำกว่าใน Churn

| กลุ่ม | freight_ratio เฉลี่ย |
|-------|---------------------|
| Churn | **0.27** |
| Stay  | **0.42** |

**ดูเหมือนค่าส่งถูก = churn มากกว่า — แต่จริงๆ คือ:**  
Churner ซื้อของแพง (R$141) → ค่าส่งเป็น % น้อยโดยอัตโนมัติ  
ตัวขับเคลื่อนจริงคือ `price` ไม่ใช่ ratio

### 7.2 `is_low_score` ต่ำกว่าใน Churn

| กลุ่ม | is_low_score (≤2 ดาว) | is_high_score (5 ดาว) |
|-------|----------------------|----------------------|
| Churn | **7.4%** | **62.7%** |
| Stay  | **32.5%** | 43.1% |

**62.7% ของ Churner ให้ 5 ดาวก่อนหายไป**  
→ Satisfaction ≠ Retention  
→ Stay customer ซื้อบ่อยจึงเจอประสบการณ์แย่บ้าง แต่ยังอยู่  
→ **ห้ามใช้ review score เป็น proxy ของ loyalty**

### 7.3 `avg_purchase_gap` น้อยใน Churn

| กลุ่ม | avg_purchase_gap |
|-------|----------------|
| Churn | 4.27 วัน |
| Stay  | 2.90 วัน |

**ดูเหมือน gap น้อย = churn — แต่จริงๆ คือ:**  
98% ของ Churner เป็น first-time buyer → gap = 0 (ยังไม่มีประวัติซื้อซ้ำ)  
ค่า 4.27 เกิดจากการ fillna ด้วย median  
feature `gap_real` แก้ปัญหานี้โดย zero-out first-time buyers

---

## 8 · Limitations

| ข้อจำกัด | ผลกระทบ | วิธีบรรเทา |
|---------|---------|-----------|
| **Predictive ≠ Causal** | Simulation ใน Action Plan เป็นการประมาณ ไม่ใช่ guarantee | ใช้เป็น directional guide เท่านั้น |
| **Class imbalance 84:16** | Model bias ไปทาง Churn ตามธรรมชาติ | Sample weighting + threshold tuning |
| **Historical data (2016–2018)** | อาจไม่ generalise กับพฤติกรรมปัจจุบัน | Retrain เมื่อมีข้อมูลใหม่ |
| **`is_extremely_late` ≈ 0% importance** | Feature นี้แทบไม่มีประโยชน์ | พิจารณาลบออกใน version ถัดไป |
| **`freight_ratio` เป็น confound ของ price** | ตีความโดดๆ mislead ได้ | ใช้คู่กับ `price` เสมอ |
| **First-time buyer dominate churn (98%)** | Model ดีกับ FTB แต่อาจพลาด nuance ของ repeat buyers | พิจารณา train แยก 2 โมเดล |

---

## 9 · Reproducibility

| Component | Detail |
|-----------|--------|
| Model file | `olist_churn_model_final (1).pkl` |
| Features file | `model_features_final (1).pkl` |
| SHAP extraction | `model.model.estimator` (unwrap FLAML → LGBMClassifier) |
| Random seed | 42 |
| Python | 3.10 |
| Key packages | `flaml`, `lightgbm`, `shap`, `scikit-learn` |
| Data source | BigQuery: `academic-moon-483615-t2.Dashboard.input` |
| Churn window | 180 วัน |
| Threshold | 0.55 |

---

*สำหรับรายละเอียด Dashboard แต่ละหน้า → ดูที่ [DASHBOARD.md](DASHBOARD.md)*
