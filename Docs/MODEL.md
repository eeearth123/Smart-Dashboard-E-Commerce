# ============================================================
# Olist Churn Model — SHAP Explainability
# วิธีใช้: อัปโหลดไฟล์นี้ใน Google Colab แล้วรัน
# ============================================================

# %% [1] Install & Import
# !pip install shap -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
sns.set_style('whitegrid')
shap.initjs()
print('✅ Ready')


# %% [2] Load Model + Data
model    = joblib.load('olist_churn_model_final (1).pkl')
features = joblib.load('model_features_final (1).pkl')

df_raw = pd.read_csv('/content/olist_ready_for_analysis (1).csv')
print(f'Loaded: {len(df_raw):,} rows')
print(f'Features ({len(features)}): {features}')


# %% [3] Feature Engineering
def process_features(df_raw):
    df = df_raw.copy()

    # แปลงวันที่
    for col in ['order_purchase_timestamp',
                'order_delivered_customer_date',
                'order_estimated_delivery_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'order_purchase_timestamp' in df.columns:
        df = df.sort_values(
            ['customer_unique_id', 'order_purchase_timestamp']
        ).reset_index(drop=True)

    # Logistics
    if 'order_delivered_customer_date' in df.columns:
        df['delivery_days'] = (
            df['order_delivered_customer_date'] -
            df['order_purchase_timestamp']
        ).dt.days.clip(lower=0)
    else:
        df['delivery_days'] = np.nan

    if 'order_estimated_delivery_date' in df.columns:
        df['estimated_days'] = (
            df['order_estimated_delivery_date'] -
            df['order_purchase_timestamp']
        ).dt.days
    else:
        df['estimated_days'] = np.nan

    df['delivery_vs_estimated'] = df['estimated_days'] - df['delivery_days']

    # Price & Freight
    if 'freight_value' in df.columns and 'price' in df.columns:
        df['freight_ratio'] = np.where(
            df['price'] > 0, df['freight_value'] / df['price'], 0)
        df['payment_value'] = df['price'] + df['freight_value']
    else:
        df['freight_ratio'] = 0
        df['payment_value'] = df.get('price', 0)

    # Review Score
    if 'review_score' in df.columns:
        df['review_score']  = pd.to_numeric(df['review_score'], errors='coerce')
        df['is_low_score']  = (df['review_score'].fillna(3) <= 2).astype(int)
        df['is_high_score'] = (df['review_score'].fillna(3) == 5).astype(int)
    else:
        df['is_low_score'] = 0
        df['is_high_score'] = 0

    # Purchase Count
    df['purchase_count']    = df.groupby('customer_unique_id').cumcount() + 1
    df['is_first_purchase'] = (df['purchase_count'] == 1).astype(int)
    df['is_repeat_buyer']   = (df['purchase_count'] >= 2).astype(int)

    # Gap Features
    if 'order_purchase_timestamp' in df.columns:
        df['prev_ts'] = df.groupby('customer_unique_id')[
            'order_purchase_timestamp'].shift(1)
        df['days_since_last_purchase'] = (
            df['order_purchase_timestamp'] - df['prev_ts']
        ).dt.days

        med_gap = df.loc[
            df['is_repeat_buyer'] == 1, 'days_since_last_purchase'
        ].median()
        if pd.isna(med_gap):
            med_gap = 90.0

        df['avg_purchase_gap'] = df.groupby('customer_unique_id')[
            'days_since_last_purchase'].transform('mean')
        df['avg_purchase_gap'] = df['avg_purchase_gap'].fillna(
            df['avg_purchase_gap'].median())
        df['gap_vs_avg'] = df['avg_purchase_gap'] - df['days_since_last_purchase']
        df['days_since_last_purchase'] = df['days_since_last_purchase'].fillna(med_gap)
        df['gap_vs_avg'] = df['gap_vs_avg'].fillna(0)
        df['gap_real']        = np.where(df['is_repeat_buyer'] == 1,
                                          df['days_since_last_purchase'], 0)
        df['gap_vs_avg_real'] = np.where(df['is_repeat_buyer'] == 1,
                                          df['gap_vs_avg'], 0)
    else:
        for c in ['days_since_last_purchase', 'avg_purchase_gap',
                  'gap_vs_avg', 'gap_real', 'gap_vs_avg_real']:
            df[c] = 0

    # 3 Features ที่โมเดลต้องการแต่ต้องคำนวณเพิ่ม
    df['first_purchase_late'] = np.where(
        df['is_first_purchase'] == 1,
        (df['delivery_vs_estimated'].fillna(0) < 0).astype(int), 0)

    df['first_purchase_bad_review'] = np.where(
        df['is_first_purchase'] == 1, df['is_low_score'], 0)

    df['is_extremely_late'] = (
        df['delivery_vs_estimated'].fillna(0) < -7).astype(int)

    return df


df = process_features(df_raw)

# เตรียม X
X    = df.reindex(columns=features, fill_value=0).fillna(0)

# ทำนาย
THRESHOLD = 0.55
proba = model.predict_proba(X)[:, 1]
pred  = (proba >= THRESHOLD).astype(int)
df['churn_probability'] = proba
df['churn_prediction']  = pred

print(f'Churn : {pred.sum():,} ({pred.mean()*100:.1f}%)')
print(f'Stay  : {(1-pred).sum():,} ({(1-pred).mean()*100:.1f}%)')


# ============================================================
# PART 1: Descriptive — Churn vs Stay ต่างกันยังไง?
# ============================================================

# %% [4] Feature Comparison Table
churn_df = df[df['churn_prediction'] == 1]
stay_df  = df[df['churn_prediction'] == 0]

display_features = [
    'price', 'freight_value', 'freight_ratio', 'delivery_days',
    'delivery_vs_estimated', 'avg_purchase_gap', 'payment_installments',
    'is_first_purchase', 'is_low_score', 'is_high_score',
    'gap_real', 'gap_vs_avg_real'
]
display_features = [f for f in display_features if f in df.columns]

compare = pd.DataFrame({
    'Stay (mean)':  stay_df[display_features].mean(),
    'Churn (mean)': churn_df[display_features].mean(),
}).round(3)
compare['Diff (Churn-Stay)'] = (
    compare['Churn (mean)'] - compare['Stay (mean)']
).round(3)
compare['Diff %'] = (
    (compare['Churn (mean)'] - compare['Stay (mean)']) /
    compare['Stay (mean)'].replace(0, np.nan) * 100
).round(1)

print('\n📊 Feature Comparison: Churn vs Stay')
print(compare.to_string())


# %% [5] Distribution Plots
top6 = ['price', 'freight_value', 'freight_ratio',
        'delivery_vs_estimated', 'avg_purchase_gap', 'delivery_days']
top6 = [f for f in top6 if f in df.columns]

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()

for i, feat in enumerate(top6):
    ax = axes[i]
    p01, p99 = df[feat].quantile([0.01, 0.99])
    v_stay  = stay_df[feat].dropna().clip(p01, p99)
    v_churn = churn_df[feat].dropna().clip(p01, p99)

    ax.hist(v_stay,  bins=40, alpha=0.6, color='#2ecc71',
            label=f'Stay  n={len(v_stay):,}', density=True, edgecolor='none')
    ax.hist(v_churn, bins=40, alpha=0.6, color='#e74c3c',
            label=f'Churn n={len(v_churn):,}', density=True, edgecolor='none')
    ax.axvline(v_stay.mean(),  color='#27ae60', linestyle='--', lw=2,
               label=f'Stay avg: {v_stay.mean():.2f}')
    ax.axvline(v_churn.mean(), color='#c0392b', linestyle='--', lw=2,
               label=f'Churn avg: {v_churn.mean():.2f}')
    ax.set_title(feat, fontweight='bold')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

plt.suptitle('Feature Distribution: Churn (red) vs Stay (green)',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('dist_churn_vs_stay.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: dist_churn_vs_stay.png')


# ============================================================
# PART 2: SHAP — Model อธิบายอะไร?
# ============================================================

# %% [6] Compute SHAP Values
np.random.seed(42)
n_sample    = min(3000, len(X))
sample_idx  = np.random.choice(len(X), size=n_sample, replace=False)
X_sample    = X.iloc[sample_idx].copy()
label_sample = pred[sample_idx]
proba_sample = proba[sample_idx]

# FLAML AutoML ห่อ estimator ไว้ข้างใน → ต้องดึงออกก่อน
if hasattr(model, 'model'):
    # flaml.AutoML → .model คือ best estimator (LGBMClassifier)
    inner_model = model.model.estimator
elif hasattr(model, 'estimator'):
    inner_model = model.estimator
else:
    inner_model = model  # fallback: ใช้ตรงๆ

print(f'Inner model type: {type(inner_model)}')

explainer   = shap.TreeExplainer(inner_model)
shap_values = explainer.shap_values(X_sample)

# LightGBM returns list [stay_shap, churn_shap]
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# base_value อาจเป็น float หรือ list ขึ้นกับเวอร์ชัน shap
base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val = base_val[1]

print(f'SHAP shape: {sv.shape}')
print(f'Base value (expected churn prob): {base_val:.3f}')


# %% [7] Beeswarm Plot
plt.figure()
shap.summary_plot(sv, X_sample, feature_names=features,
                  plot_type='dot', max_display=15, show=False)
plt.title('Beeswarm — แดง=ค่าสูง/น้ำเงิน=ค่าต่ำ | ขวา=เพิ่ม churn/ซ้าย=ลด churn',
          fontsize=10)
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: shap_beeswarm.png')


# %% [8] Bar Plot
plt.figure()
shap.summary_plot(sv, X_sample, feature_names=features,
                  plot_type='bar', max_display=15, show=False)
plt.title('Feature Importance — Mean |SHAP|', fontsize=11)
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: shap_bar.png')


# %% [9] Churn vs Stay Split
churn_mask_s = label_sample == 1
stay_mask_s  = label_sample == 0

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

plt.sca(axes[0])
shap.summary_plot(sv[churn_mask_s], X_sample[churn_mask_s],
                  feature_names=features, max_display=12,
                  plot_type='dot', show=False, color_bar=False)
axes[0].set_title(f'CHURN (n={churn_mask_s.sum():,}) — features ที่ดัน risk สูง',
                  fontweight='bold', color='#c0392b')

plt.sca(axes[1])
shap.summary_plot(sv[stay_mask_s], X_sample[stay_mask_s],
                  feature_names=features, max_display=12,
                  plot_type='dot', show=False, color_bar=True)
axes[1].set_title(f'STAY (n={stay_mask_s.sum():,}) — features ที่กด risk ลง',
                  fontweight='bold', color='#27ae60')

plt.suptitle('SHAP Values: Churn vs Stay แยกกัน', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_churn_vs_stay.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: shap_churn_vs_stay.png')


# %% [10] Dependence Plots
top4 = ['freight_value', 'price', 'delivery_vs_estimated', 'avg_purchase_gap']
top4 = [f for f in top4 if f in features]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for i, feat in enumerate(top4):
    ax        = axes[i]
    feat_idx  = list(features).index(feat)
    x_vals    = X_sample[feat].values
    y_vals    = sv[:, feat_idx]
    p1, p99   = np.percentile(x_vals, [1, 99])
    mask      = (x_vals >= p1) & (x_vals <= p99)

    sc = ax.scatter(x_vals[mask], y_vals[mask],
                    c=y_vals[mask], cmap='RdYlGn_r',
                    alpha=0.4, s=8, vmin=-0.3, vmax=0.3)
    ax.axhline(0, color='black', lw=1, linestyle='--', alpha=0.5)

    z    = np.polyfit(x_vals[mask], y_vals[mask], 1)
    xln  = np.linspace(x_vals[mask].min(), x_vals[mask].max(), 100)
    ax.plot(xln, np.poly1d(z)(xln), color='#2c3e50', lw=2, alpha=0.8)

    ax.set_xlabel(feat)
    ax.set_ylabel('SHAP value (บวก = เพิ่ม churn)')
    ax.set_title(f'{feat} — ค่าเท่าไหร่เริ่ม trigger churn?', fontweight='bold')
    plt.colorbar(sc, ax=ax, label='SHAP')

plt.suptitle('Dependence Plots — ความสัมพันธ์ค่า Feature vs Churn Risk',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('shap_dependence.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: shap_dependence.png')


# %% [11] Group Mean SHAP Bar
sv_churn_mean = sv[churn_mask_s].mean(axis=0)
sv_stay_mean  = sv[stay_mask_s].mean(axis=0)

shap_compare = pd.DataFrame({
    'Feature':    features,
    'SHAP_Churn': sv_churn_mean,
    'SHAP_Stay':  sv_stay_mean,
}).sort_values('SHAP_Churn', ascending=False)

x = np.arange(len(shap_compare))
w = 0.35
fig, ax = plt.subplots(figsize=(14, 7))
ax.bar(x - w/2, shap_compare['SHAP_Churn'], w, color='#e74c3c',
       alpha=0.8, label='Churn', edgecolor='white')
ax.bar(x + w/2, shap_compare['SHAP_Stay'],  w, color='#2ecc71',
       alpha=0.8, label='Stay',  edgecolor='white')
ax.axhline(0, color='black', lw=1)
ax.set_xticks(x)
ax.set_xticklabels(shap_compare['Feature'], rotation=45, ha='right')
ax.set_ylabel('Mean SHAP (บวก = ดัน churn | ลบ = กด churn)')
ax.set_title('Churn vs Stay — Mean SHAP per Feature', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('shap_group_compare.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: shap_group_compare.png')


# ============================================================
# PART 3: รายคน — Waterfall
# ============================================================

# %% [12] Waterfall: 4 กรณี
from shap import Explanation

shap_exp = Explanation(
    values        = sv,
    base_values   = np.full(len(X_sample), base_val),
    data          = X_sample.values,
    feature_names = list(features)
)

cases = {
    'High Risk Churn\n(prob > 0.9)':      np.where(proba_sample > 0.90)[0],
    'Borderline Churn\n(0.55-0.65)':      np.where((proba_sample >= 0.55) &
                                                     (proba_sample < 0.65))[0],
    'Borderline Stay\n(0.45-0.55)':       np.where((proba_sample >= 0.45) &
                                                     (proba_sample < 0.55))[0],
    'Safe Stay\n(prob < 0.2)':            np.where(proba_sample < 0.20)[0],
}

fig, axes = plt.subplots(2, 2, figsize=(22, 14))
axes = axes.flatten()

for i, (title, idx_pool) in enumerate(cases.items()):
    ax = axes[i]
    if len(idx_pool) == 0:
        ax.text(0.5, 0.5, 'No samples', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        continue
    pick = idx_pool[len(idx_pool) // 2]
    plt.sca(ax)
    shap.waterfall_plot(shap_exp[pick], max_display=10, show=False)
    color = '#c0392b' if proba_sample[pick] >= THRESHOLD else '#27ae60'
    ax.set_title(f'{title}\nchurn_prob = {proba_sample[pick]:.3f}',
                 fontweight='bold', color=color)

plt.suptitle('Waterfall — ทำไมคนนี้ถึงได้ churn score นี้?\n'
             'แดง = ดัน churn สูง | น้ำเงิน = กด churn ลง',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: shap_waterfall.png')


# ============================================================
# PART 4: Segment Profile Heatmap
# ============================================================

# %% [13] Segment Heatmap
df['risk_group'] = pd.cut(
    df['churn_probability'],
    bins=[0, 0.4, 0.55, 0.75, 1.0],
    labels=['Safe < 0.4', 'Medium 0.4-0.55',
            'High 0.55-0.75', 'Very High > 0.75']
)

profile_feats = ['price', 'freight_value', 'freight_ratio', 'delivery_days',
                 'delivery_vs_estimated', 'avg_purchase_gap',
                 'is_first_purchase', 'is_low_score', 'gap_real']
profile_feats = [f for f in profile_feats if f in df.columns]

profile = df.groupby('risk_group', observed=True)[profile_feats].mean().round(3)

# normalize 0-1 per feature
profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(
    profile_norm.T,
    annot=profile.T,
    fmt='.2f',
    cmap='RdYlGn_r',
    ax=ax,
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Normalized (0=low, 1=high)'}
)
ax.set_title('Feature Profile per Risk Group\n'
             '(ตัวเลข = ค่าจริง | สี = relative level)',
             fontweight='bold', fontsize=13)
ax.set_xlabel('Risk Group')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig('segment_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: segment_heatmap.png')

print('\n📊 Segment sizes:')
print(df['risk_group'].value_counts().sort_index())


# ============================================================
# FINAL SUMMARY
# ============================================================

# %% [14] Summary
shap_importance = pd.DataFrame({
    'Feature':    list(features),
    'Mean|SHAP|': np.abs(sv).mean(axis=0),
    'SHAP_Churn': sv[churn_mask_s].mean(axis=0),
    'SHAP_Stay':  sv[stay_mask_s].mean(axis=0),
}).sort_values('Mean|SHAP|', ascending=False)

print('=' * 65)
print('FINAL SUMMARY — พฤติกรรม Churn vs Stay')
print('=' * 65)
print()
print('CHURN มีลักษณะ (features ที่ดัน churn สูง):')
for _, r in shap_importance[shap_importance['SHAP_Churn'] > 0].head(6).iterrows():
    c_avg = churn_df[r['Feature']].mean() if r['Feature'] in churn_df.columns else 0
    s_avg = stay_df[r['Feature']].mean()  if r['Feature'] in stay_df.columns  else 0
    print(f"  {r['Feature']:30s}  churn={c_avg:7.2f}  stay={s_avg:7.2f}  "
          f"SHAP={r['SHAP_Churn']:+.3f}")

print()
print('STAY ถูกปกป้องโดย (features ที่กด churn ลง):')
for _, r in shap_importance[shap_importance['SHAP_Stay'] < 0].head(4).iterrows():
    c_avg = churn_df[r['Feature']].mean() if r['Feature'] in churn_df.columns else 0
    s_avg = stay_df[r['Feature']].mean()  if r['Feature'] in stay_df.columns  else 0
    print(f"  {r['Feature']:30s}  churn={c_avg:7.2f}  stay={s_avg:7.2f}  "
          f"SHAP={r['SHAP_Stay']:+.3f}")

print()
print('Output files:')
files = ['dist_churn_vs_stay.png', 'shap_beeswarm.png', 'shap_bar.png',
         'shap_churn_vs_stay.png', 'shap_dependence.png',
         'shap_group_compare.png', 'shap_waterfall.png', 'segment_heatmap.png']
for f in files:
    print(f'  {f}')
print('=' * 65)
