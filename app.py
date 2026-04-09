import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP
# ==========================================
st.set_page_config(
    page_title="Olist Executive Cockpit",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .metric-card { background-color:#f0f2f6; border-radius:10px;
                   padding:15px; box-shadow:2px 2px 5px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD FROM BIGQUERY
# ==========================================
from google.oauth2 import service_account
from google.cloud import bigquery

@st.cache_data(ttl=600)
def load_bq_data():
    try:
        creds_info = st.secrets["connections"]["bigquery"]["service_account_info"]
        scopes = [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/bigquery"
        ]
        credentials = service_account.Credentials.from_service_account_info(
            creds_info, scopes=scopes)
        client = bigquery.Client(
            credentials=credentials,
            project=creds_info["project_id"],
            location="asia-southeast1"
        )
        df = client.query(
            "SELECT * FROM `academic-moon-483615-t2.Dashboard.input`"
        ).to_dataframe()
        return df, None
    except Exception as e:
        return None, str(e)

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
def process_features(df_raw):
    df = df_raw.copy()

    for col in ['order_purchase_timestamp',
                'order_delivered_customer_date',
                'order_estimated_delivery_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    if 'order_purchase_timestamp' in df.columns:
        df = df.sort_values(['customer_unique_id', 'order_purchase_timestamp']
                            ).reset_index(drop=True)

    # Logistics
    if 'order_delivered_customer_date' in df.columns and \
       'order_purchase_timestamp' in df.columns:
        df['delivery_days'] = (
            df['order_delivered_customer_date'] -
            df['order_purchase_timestamp']
        ).dt.days.clip(lower=0)
    else:
        df['delivery_days'] = np.nan

    if 'order_estimated_delivery_date' in df.columns and \
       'order_purchase_timestamp' in df.columns:
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

    # Payment Features
    if 'payment_sequential' in df.columns:
        df['uses_multiple_payments'] = (
            df['payment_sequential'].fillna(1) > 1).astype(int)
    else:
        df['uses_multiple_payments'] = 0

    if 'payment_type' in df.columns:
        df['uses_voucher'] = (
            df['payment_type'].fillna('') == 'voucher').astype(int)
    else:
        df['uses_voucher'] = 0

    # Review Score
    if 'review_score' in df.columns:
        df['review_score']  = pd.to_numeric(df['review_score'], errors='coerce')
        df['is_low_score']  = (df['review_score'].fillna(3) <= 2).astype(int)
        df['is_high_score'] = (df['review_score'].fillna(3) == 5).astype(int)
    else:
        df['review_score']  = 3.0
        df['is_low_score']  = 0
        df['is_high_score'] = 0

    # Purchase Count & Repeat
    df['purchase_count']    = df.groupby('customer_unique_id').cumcount() + 1
    df['is_first_purchase'] = (df['purchase_count'] == 1).astype(int)
    df['is_repeat_buyer']   = (df['purchase_count'] >= 2).astype(int)

    # Gap Features
    if 'order_purchase_timestamp' in df.columns:
        df['prev_purchase_date'] = df.groupby('customer_unique_id')[
            'order_purchase_timestamp'].shift(1)
        df['days_since_last_purchase'] = (
            df['order_purchase_timestamp'] - df['prev_purchase_date']
        ).dt.days

        median_gap = df.loc[df['is_repeat_buyer'] == 1,
                            'days_since_last_purchase'].median()
        if pd.isna(median_gap): median_gap = 90.0

        df['avg_purchase_gap'] = (
            df.groupby('customer_unique_id')['days_since_last_purchase']
            .transform('mean')
        )
        global_avg = df['avg_purchase_gap'].median()
        df['avg_purchase_gap'] = df['avg_purchase_gap'].fillna(global_avg)
        df['gap_vs_avg'] = df['avg_purchase_gap'] - df['days_since_last_purchase']
        df['days_since_last_purchase'] = df['days_since_last_purchase'].fillna(median_gap)
        df['gap_vs_avg'] = df['gap_vs_avg'].fillna(0)
        df['gap_real']        = np.where(df['is_repeat_buyer'] == 1,
                                         df['days_since_last_purchase'], 0)
        df['gap_vs_avg_real'] = np.where(df['is_repeat_buyer'] == 1,
                                         df['gap_vs_avg'], 0)
    else:
        for c in ['days_since_last_purchase', 'avg_purchase_gap',
                  'gap_vs_avg', 'gap_real', 'gap_vs_avg_real']:
            df[c] = 0

    # cat_churn_risk placeholder (will be overwritten after predict)
    if 'cat_churn_risk' not in df.columns:
        df['cat_churn_risk'] = 0.80

    # Lateness Score
    if 'order_purchase_timestamp' in df.columns:
        ref_date   = df['order_purchase_timestamp'].max()
        last_order = df.groupby('customer_unique_id')[
            'order_purchase_timestamp'].transform('max')
        df['days_since_purchase'] = (ref_date - last_order).dt.days

        tmp = df.sort_values(['customer_unique_id', 'product_category_name',
                              'order_purchase_timestamp'])
        tmp['prev_ts'] = tmp.groupby(
            ['customer_unique_id', 'product_category_name']
        )['order_purchase_timestamp'].shift(1)
        tmp['order_gap'] = (
            tmp['order_purchase_timestamp'] - tmp['prev_ts']
        ).dt.days
        valid_gaps = tmp[(tmp['order_gap'] >= 7) & (tmp['order_gap'] <= 730)]
        if len(valid_gaps) > 10:
            cat_med = valid_gaps.groupby('product_category_name')[
                'order_gap'].median().rename('cat_median_days')
            df = df.merge(cat_med, on='product_category_name', how='left')
        else:
            df['cat_median_days'] = 180
        df['cat_median_days'] = df['cat_median_days'].fillna(180).clip(lower=7)
        df['lateness_score']  = (
            df['days_since_purchase'] / df['cat_median_days']
        ).clip(lower=0)
    else:
        df['days_since_purchase'] = 90
        df['cat_median_days']     = 180
        df['lateness_score']      = 0.5

    # delay_days
    if 'order_delivered_customer_date' in df.columns and \
       'order_estimated_delivery_date' in df.columns:
        df['delay_days'] = (
            df['order_delivered_customer_date'] -
            df['order_estimated_delivery_date']
        ).dt.days.fillna(0)
    else:
        df['delay_days'] = 0

    return df

# ==========================================
# 4. LOAD MODEL
# ==========================================
@st.cache_resource
def load_models():
    d = os.path.dirname(os.path.abspath(__file__))
    try:
        model    = joblib.load(os.path.join(d, 'olist_churn_model_final (1).pkl'))
        features = joblib.load(os.path.join(d, 'model_features_final (1).pkl'))
        return model, features, None
    except Exception as e:
        return None, None, str(e)

# ==========================================
# 5. PREDICT
# ==========================================
def predict_churn(df, model, feature_names, threshold):
    X = pd.DataFrame(index=df.index)
    for col in feature_names:
        X[col] = df[col] if col in df.columns else 0
    X = X.fillna(X.median())
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X).astype(float)
    return proba, (proba >= threshold).astype(int)

# ==========================================
# 6. SIDEBAR & LOAD
# ==========================================
with st.sidebar:
    if st.button('🔄 Refresh Data'):
        st.cache_data.clear()
        st.rerun()

df_raw, bq_error = load_bq_data()
model, feature_names, model_error = load_models()
best_threshold = 0.55

if bq_error:
    st.error(f"⚠️ BigQuery Error: {bq_error}")
    st.stop()
if model_error:
    st.warning(f"⚠️ Model: {model_error}")

# ==========================================
# 7. PROCESS & PREDICT
# ==========================================
df = process_features(df_raw)

if model is not None and feature_names:
    proba, pred = predict_churn(df, model, feature_names, best_threshold)
    df['churn_probability'] = proba
    df['churn_prediction']  = pred
    if 'product_category_name' in df.columns:
        cat_risk_map = df.groupby('product_category_name')['churn_probability'].mean()
        df['cat_churn_risk'] = df['product_category_name'].map(cat_risk_map)
else:
    df['churn_probability'] = 0.5
    df['churn_prediction']  = 1

df['is_churn'] = df['churn_prediction']

# ==========================================
# 8. STATUS
# ==========================================
def get_status(row):
    prob = row.get('churn_probability', 0)
    late = row.get('lateness_score', 0)
    if late > 3.0:   return 'Lost (Late > 3x)'
    if prob > 0.75:  return 'High Risk'
    if late > 1.5:   return 'Warning (Late > 1.5x)'
    if prob >= 0.40: return 'Medium Risk'
    return 'Active'

df['status'] = df.apply(get_status, axis=1)

# ==========================================
# 9. NAVIGATION
# ==========================================
st.sidebar.title("✈️ Olist Cockpit")
st.sidebar.success(f"✅ โหลดข้อมูลแล้ว ({len(df):,} rows)")
st.sidebar.info(f"🎯 Model Threshold: {best_threshold:.2f}")
st.sidebar.markdown("""
**📊 Business Rules:**
- 🔴 Lost: Late > 3.0
- 🟥 High Risk: AI > 75%
- 🟧 Warning: Late > 1.5
- 🟨 Medium Risk: AI 40–75%
- 🟩 Active: AI < 40%
""")
page = st.sidebar.radio("Navigation", [
    "1. 💰 Business Overview",
    "2. 📊 Churn Overview",
    "3. 🎯 Action Plan",
    "4. 🔄 Buying Cycle Analysis",
    "5. 🚛 Logistics Insights",
    "6. 🏪 Seller Audit",
    "7. 🔍 Customer Detail",
])
st.sidebar.markdown("---")

# helper
def safe_cats(dataframe, col='product_category_name'):
    if col not in dataframe.columns: return []
    return sorted([x for x in dataframe[col].unique() if pd.notna(x)])

# ==========================================
# PAGE 1: Business Overview
# ==========================================
if page == "1. 💰 Business Overview":
    st.title("💰 Business Overview")
    st.caption("ภาพรวมรายได้และสุขภาพธุรกิจ")

    with st.expander("🌪️ กรองข้อมูล", expanded=False):
        sel_cats = st.multiselect("หมวดสินค้า (ว่าง = ทั้งหมด):", safe_cats(df), key="p1_cat")

    df_d = df[df['product_category_name'].isin(sel_cats)].copy() if sel_cats else df.copy()
    st.markdown("---")

    total_rev   = df_d['payment_value'].sum() if 'payment_value' in df_d.columns else 0
    avg_order   = df_d['payment_value'].mean() if 'payment_value' in df_d.columns else 0
    n_customers = df_d['customer_unique_id'].nunique() if 'customer_unique_id' in df_d.columns else 0
    clv         = avg_order * df_d.groupby('customer_unique_id').size().mean() \
                  if n_customers > 0 else avg_order

    mom_growth = None
    if 'order_purchase_timestamp' in df_d.columns and not df_d.empty:
        df_d['_month'] = df_d['order_purchase_timestamp'].dt.to_period('M')
        all_months = pd.period_range(start=df_d['_month'].min(),
                                     end=df_d['_month'].max(), freq='M')
        monthly_rev = df_d.groupby('_month')['payment_value'].sum().reindex(all_months, fill_value=0)
        if len(monthly_rev) >= 3:
            last_m, prev_m = monthly_rev.iloc[-2], monthly_rev.iloc[-3]
            if prev_m > 0: mom_growth = (last_m - prev_m) / prev_m * 100
        elif len(monthly_rev) == 2:
            last_m, first_m = monthly_rev.iloc[-1], monthly_rev.iloc[-2]
            if first_m > 0: mom_growth = (last_m - first_m) / first_m * 100

    k1, k2, k3 = st.columns(3)
    k1.metric("💰 Total Revenue",   f"R$ {total_rev:,.0f}")
    k2.metric("📈 MoM Growth",      f"{mom_growth:+.1f}%" if mom_growth is not None else "N/A",
              delta=f"{mom_growth:+.1f}%" if mom_growth is not None else None)
    k3.metric("🛒 Avg Order Value", f"R$ {avg_order:,.0f}")
    st.markdown("---")

    st.subheader("📈 Monthly Revenue Trend")
    if 'order_purchase_timestamp' in df_d.columns and not df_d.empty:
        rev_trend = (df_d.set_index('order_purchase_timestamp')['payment_value']
                     .resample('MS').sum().fillna(0).reset_index())
        rev_trend.columns = ['Month', 'Revenue']
        rev_trend['Growth'] = (rev_trend['Revenue'].pct_change()
                               .replace([np.inf, -np.inf], np.nan) * 100)
        plot_df = rev_trend.iloc[:-1] if len(rev_trend) > 1 else rev_trend

        base  = alt.Chart(plot_df).encode(
            x=alt.X('Month:T', axis=alt.Axis(format='%b %Y', labelAngle=-45, title='')))
        bars  = base.mark_bar(color='#1E88E5', opacity=0.7).encode(
            y=alt.Y('Revenue:Q', title='Revenue (R$)', axis=alt.Axis(grid=False)),
            tooltip=[alt.Tooltip('Month:T', format='%B %Y'),
                     alt.Tooltip('Revenue:Q', format=',.0f')])
        line  = base.mark_line(color='#E53935', strokeWidth=3,
                               point=alt.OverlayMarkDef(color='#E53935')).encode(
            y=alt.Y('Growth:Q', title='Growth (%)',
                    axis=alt.Axis(titleColor='#E53935', orient='right')),
            tooltip=[alt.Tooltip('Month:T', format='%B %Y'),
                     alt.Tooltip('Growth:Q', format='.1f', title='Growth %')])
        st.altair_chart(alt.layer(bars, line).resolve_scale(y='independent')
                        .properties(height=350), use_container_width=True)
    else:
        st.info("ไม่มีข้อมูลเพียงพอ")

    # ── หมวดสินค้าขายดี ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏆 หมวดสินค้าขายดี (Top Categories)")
    if 'product_category_name' in df_d.columns and not df_d.empty:
        cat_sales = df_d.groupby('product_category_name').agg(
            revenue=('payment_value', 'sum'),
            orders=('payment_value', 'count'),
            avg_order=('payment_value', 'mean'),
            churn_risk=('churn_probability', 'mean')
        ).reset_index().sort_values('revenue', ascending=False)

        col_chart, col_table = st.columns([1.5, 2])

        with col_chart:
            top20 = cat_sales.head(20)
            bar_cat = alt.Chart(top20).mark_bar().encode(
                x=alt.X('revenue:Q', title='Revenue (R$)'),
                y=alt.Y('product_category_name:N', sort='-x', title=None),
                color=alt.Color('churn_risk:Q',
                    scale=alt.Scale(domain=[0.3, 0.9], range=['#2ecc71', '#e74c3c']),
                    title='Churn Risk'),
                tooltip=[
                    alt.Tooltip('product_category_name', title='หมวด'),
                    alt.Tooltip('revenue', format=',.0f', title='Revenue (R$)'),
                    alt.Tooltip('orders', format=',', title='จำนวน Orders'),
                    alt.Tooltip('churn_risk', format='.1%', title='Churn Risk'),
                ]
            ).properties(height=500, title='Top 20 หมวดสินค้า (สีแดง = Churn Risk สูง)')
            st.altair_chart(bar_cat, use_container_width=True)

        with col_table:
            st.markdown("**📋 รายละเอียดทุกหมวด**")
            st.dataframe(
                cat_sales.rename(columns={
                    'product_category_name': 'หมวดสินค้า',
                    'revenue':   'Revenue (R$)',
                    'orders':    'Orders',
                    'avg_order': 'Avg Order (R$)',
                    'churn_risk':'Churn Risk'
                }),
                column_config={
                    'Revenue (R$)':    st.column_config.NumberColumn(format='R$ %.0f'),
                    'Orders':          st.column_config.NumberColumn(format='%,d'),
                    'Avg Order (R$)':  st.column_config.NumberColumn(format='R$ %.0f'),
                    'Churn Risk':      st.column_config.ProgressColumn(
                        format='%.2f', min_value=0, max_value=1),
                },
                use_container_width=True,
                hide_index=True,
                height=500
            )

# ==========================================
# PAGE 2: Churn Overview
# ==========================================
elif page == "2. 📊 Churn Overview":
    st.title("📊 Churn Overview")

    with st.expander("ℹ️ วิธีแบ่งกลุ่มลูกค้า", expanded=True):
        st.markdown("""
| สถานะ | เงื่อนไข |
|---|---|
| 🔴 Lost | Lateness > 3.0 |
| 🟥 High Risk | AI > 75% |
| 🟧 Warning | Lateness > 1.5 |
| 🟨 **Medium Risk** | **AI 40–75%** (ขยายแล้ว!) |
| 🟩 Active | AI < 40% |
        """)

    with st.expander("🌪️ กรองข้อมูล", expanded=False):
        sel_cats = st.multiselect("หมวดสินค้า:", safe_cats(df), key="p2_cat")

    df_d = df[df['product_category_name'].isin(sel_cats)].copy() if sel_cats else df.copy()
    st.markdown("---")

    total   = len(df_d)
    risk_df = df_d[df_d['status'].isin(['High Risk', 'Warning (Late > 1.5x)'])]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("🚨 At-Risk",         f"{len(risk_df)/total*100:.1f}%" if total else "0%")
    k2.metric("🤖 AI Predicted",    f"{(df_d['churn_probability'] >= best_threshold).mean()*100:.1f}%")
    k3.metric("💸 Revenue at Risk", f"R$ {risk_df['payment_value'].sum():,.0f}")
    k4.metric("👥 Risk / Total",    f"{len(risk_df):,} / {total:,}")
    k5.metric("🔄 Avg Cycle",       f"{df_d['cat_median_days'].mean():.0f} วัน"
                                    if 'cat_median_days' in df_d.columns else "N/A")
    st.markdown("---")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📈 Churn Risk Trend")
        if 'order_purchase_timestamp' in df_d.columns and not df_d.empty:
            df_d['month_year'] = df_d['order_purchase_timestamp'].dt.to_period('M')
            trend_data = []
            for name, grp in df_d.groupby('month_year'):
                t = len(grp)
                if t == 0: continue
                rule = len(grp[grp['status'].isin(['High Risk', 'Warning (Late > 1.5x)'])])
                ai   = (grp['churn_probability'] >= best_threshold).sum()
                trend_data.append({'Date': str(name),
                                   'Rule-based (%)': rule/t*100,
                                   'AI Predicted (%)': ai/t*100})
            tdf = pd.DataFrame(trend_data)
            if len(tdf) > 1:
                tdf = tdf.iloc[:-1]
                tdf['Date'] = pd.to_datetime(tdf['Date'])
                melted = tdf.melt('Date', var_name='Type', value_name='Rate (%)')
                chart = alt.Chart(melted).mark_line(point=True).encode(
                    x=alt.X('Date', axis=alt.Axis(format='%b %Y', title='Timeline')),
                    y=alt.Y('Rate (%)', title='Churn Rate (%)'),
                    color=alt.Color('Type', scale=alt.Scale(
                        domain=['Rule-based (%)', 'AI Predicted (%)'],
                        range=['#e67e22', '#8e44ad']),
                        legend=alt.Legend(orient='bottom')),
                    tooltip=['Date', 'Type', alt.Tooltip('Rate (%)', format='.1f')]
                ).properties(height=350)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("ข้อมูลไม่เพียงพอสำหรับ Trend")

    with c2:
        st.subheader("💰 Revenue by Risk")
        if not df_d.empty:
            stats = df_d.groupby('status').agg(
                Count=('customer_unique_id', 'count'),
                Revenue=('payment_value', 'sum')
            ).reset_index()
            domain = ['Active', 'Medium Risk', 'Warning (Late > 1.5x)', 'High Risk', 'Lost (Late > 3x)']
            range_ = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#95a5a6']
            donut = alt.Chart(stats).mark_arc(innerRadius=60).encode(
                theta=alt.Theta('Count', type='quantitative'),
                color=alt.Color('status', scale=alt.Scale(domain=domain, range=range_),
                                legend=dict(orient='bottom')),
                tooltip=['status', alt.Tooltip('Count', format=','),
                         alt.Tooltip('Revenue', format=',.0f')]
            ).properties(height=350)
            st.altair_chart(donut, use_container_width=True)

# ==========================================
# PAGE 3: Action Plan (Model-Driven)
# ==========================================
elif page == "3. 🎯 Action Plan":
    st.title("🎯 Action Plan & Simulator")
    st.caption("จำลองผลกระทบโดยเปลี่ยนฟีเจอร์ → ทำนายซ้ำด้วยโมเดล → วัด Uplift จริง")

    with st.expander("🎯 กำหนดกลุ่มเป้าหมาย", expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            risk_segments = st.multiselect(
                "กลุ่มความเสี่ยง:",
                ['High Risk', 'Warning (Late > 1.5x)', 'Medium Risk', 'Lost (Late > 3x)'],
                default=['High Risk', 'Warning (Late > 1.5x)']
            )
        with f2:
            sel_cats_p3 = st.multiselect("หมวดสินค้า (ว่าง = ทุกหมวด):",
                                         safe_cats(df), key="p3_cat_multiselect")

    df_p3 = df.copy()
    if risk_segments:
        df_p3 = df_p3[df_p3['status'].isin(risk_segments)]
    if sel_cats_p3:
        df_p3 = df_p3[df_p3['product_category_name'].isin(sel_cats_p3)]

    filter_msg = (f"กลุ่ม: {', '.join(risk_segments[:2])}"
                  f"{'...' if len(risk_segments)>2 else ''}") if risk_segments else "ทุกกลุ่ม"
    total_pop = len(df_p3)
    avg_ltv   = float(df_p3['payment_value'].mean()) if 'payment_value' in df_p3.columns else 150.0

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: st.info(f"📊 กำลังวิเคราะห์: **{filter_msg}**")
    with c2: st.metric("👥 กลุ่มเป้าหมาย", f"{total_pop:,} คน")
    with c3: st.metric("💰 LTV เฉลี่ย/คน", f"R$ {avg_ltv:,.0f}")
    st.markdown("---")

    def run_simulation(target_df, feature_changes: dict, cost_per_head: float,
                       tab_key: str, rec_text: str, strategy_name: str):
        n_target    = len(target_df)
        pct_problem = (n_target / total_pop * 100) if total_pop > 0 else 0

        c_prob, c_sol, c_res = st.columns([1, 1.3, 1])

        with c_prob:
            st.info(f"**📉 ปัญหา:** พบ {n_target:,} คน\n({pct_problem:.1f}% ของกลุ่มนี้)")
            st.progress(min(pct_problem / 100, 1.0))
            st.caption("แถบสีแสดงสัดส่วนคนที่มีปัญหา")
            if not target_df.empty:
                st.markdown("**📋 Feature เฉลี่ย:**")
                for col in list(feature_changes.keys())[:3]:
                    if col in target_df.columns:
                        st.caption(f"• {col}: {target_df[col].mean():.2f}")

        with c_sol:
            st.markdown(f"**🛠️ วิธีแก้ไข: {strategy_name}**")
            st.write(rec_text)
            st.markdown("---")
            cost = st.number_input(
                "งบต่อหัว (R$)", value=float(cost_per_head),
                min_value=0.0, max_value=500.0, step=0.5,
                key=f"cost_{tab_key}"
            )
            break_even_rate = cost / avg_ltv if avg_ltv > 0 else 0
            st.caption(f"📐 จุดคุ้มทุน: ต้องสำเร็จ ≥ **{break_even_rate:.1%}**")

            if model is None or not feature_names:
                max_pot  = 15
                realistic = min(max_pot, 10) if cost >= 15 else min(max_pot, 5)
                st.markdown(f"**🤖 AI Prediction:** `{realistic}%`")
                st.caption("(โมเดลไม่พร้อม → ใช้ค่าประมาณ)")
                lift = st.slider("ปรับค่าคาดการณ์ความสำเร็จ (%)", 1, 100,
                                 realistic, key=f"lift_{tab_key}")
                sim_success_rate = lift / 100
                sim_mode = "manual"
            else:
                sim_mode = "model"
                lift     = None

        with c_res:
            with st.spinner("⚡ โมเดลกำลังจำลอง..."):
                time.sleep(0.3)

                if sim_mode == "model" and not target_df.empty:
                    # ── ทำนายค่าเดิม ──────────────────────────────────────
                    X_orig    = target_df.reindex(columns=feature_names, fill_value=0).fillna(0)
                    prob_orig = model.predict_proba(X_orig)[:, 1]

                    # ── แก้ features ตามแคมเปญ ────────────────────────────
                    df_sim = target_df.copy()
                    for col, (op, val) in feature_changes.items():
                        if col in df_sim.columns:
                            if op == 'set':         df_sim[col] = val
                            elif op == 'multiply':  df_sim[col] = df_sim[col] * val
                            elif op == 'clip_upper':df_sim[col] = df_sim[col].clip(upper=val)
                            elif op == 'add':       df_sim[col] = df_sim[col] + val

                    # recalc freight_ratio
                    if 'freight_value' in df_sim.columns and 'price' in df_sim.columns:
                        df_sim['freight_ratio'] = (
                            df_sim['freight_value'] /
                            df_sim['price'].replace(0, np.nan)
                        ).fillna(0)

                    # ── ทำนายค่าใหม่ ──────────────────────────────────────
                    X_sim    = df_sim.reindex(columns=feature_names, fill_value=0).fillna(0)
                    prob_sim = model.predict_proba(X_sim)[:, 1]

                    # ── Uplift ────────────────────────────────────────────
                    uplift_arr       = prob_orig - prob_sim
                    THRESHOLD        = 0.08
                    sim_success_rate = (uplift_arr > THRESHOLD).mean()

                    # ── Uplift distribution chart ─────────────────────────
                    dist = {
                        "ตอบสนองสูง\n(>15%)":      int((uplift_arr > 0.15).sum()),
                        "ปานกลาง\n(8–15%)":         int(((uplift_arr > 0.08) & (uplift_arr <= 0.15)).sum()),
                        "ต่ำ\n(0–8%)":              int(((uplift_arr > 0) & (uplift_arr <= 0.08)).sum()),
                        "ไม่ตอบสนอง":               int((uplift_arr <= 0).sum()),
                    }
                    dist_df = pd.DataFrame({"กลุ่ม": list(dist.keys()),
                                            "จำนวน": list(dist.values())})
                    st.altair_chart(
                        alt.Chart(dist_df).mark_bar().encode(
                            x=alt.X("กลุ่ม", sort=None, axis=alt.Axis(labelAngle=0)),
                            y=alt.Y("จำนวน"),
                            color=alt.Color("กลุ่ม", scale=alt.Scale(
                                domain=list(dist.keys()),
                                range=["#2ecc71", "#f1c40f", "#e67e22", "#95a5a6"]
                            ), legend=None),
                            tooltip=["กลุ่ม", "จำนวน"]
                        ).properties(height=160, title="📊 Uplift Distribution"),
                        use_container_width=True
                    )
                else:
                    sim_success_rate = lift / 100 if lift else 0.1

                # ── ROI ───────────────────────────────────────────────────
                budget      = n_target * cost
                saved_users = int(n_target * sim_success_rate)
                revenue     = saved_users * avg_ltv
                profit      = revenue - budget
                roi         = (profit / budget * 100) if budget > 0 else 0
                be_final    = cost / avg_ltv if avg_ltv > 0 else 0

                st.markdown("**🚀 ผลลัพธ์**")
                st.metric("🤖 Success Rate (โมเดล)", f"{sim_success_rate:.1%}",
                          delta=f"จุดคุ้มทุน {be_final:.1%}")
                st.metric("👥 ดึงลูกค้าคืน",   f"{saved_users:,} คน")
                st.metric("💸 งบประมาณ",       f"R$ {budget:,.0f}")

                if profit > 0:
                    st.metric("📈 กำไรสุทธิ (ROI)", f"R$ {profit:,.0f}", f"+{roi:.1f}%")
                    st.success("✅ **คุ้มค่าการลงทุน!**")
                else:
                    st.metric("📉 ขาดทุนสุทธิ", f"R$ {profit:,.0f}", f"{roi:.1f}%")
                    gap = be_final - sim_success_rate
                    st.error(
                        f"⚠️ **ขาดทุน!**\n\n"
                        f"ต้องการ Success Rate: **{be_final:.1%}**\n"
                        f"ได้จริง: **{sim_success_rate:.1%}**\n"
                        f"ขาดอีก: **{gap:.1%}**"
                    )
                    max_cost_be = avg_ltv * sim_success_rate
                    st.caption(f"💡 ลดงบต่อหัวเหลือ **R$ {max_cost_be:.0f}** เพื่อเริ่มกำไร")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🚚 1. ส่งฟรี / ลดค่าส่ง",
        "💵 2. ส่วนลดสินค้า",
        "❤️ 3. ง้อลูกค้าส่งช้า",
        "🛍️ 4. ขายพ่วง / Cross-sell"
    ])

    with tab1:
        st.subheader("🚚 กลุ่มค่าส่งแพงเกินรับไหว (Freight Pain)")
        if 'freight_ratio' in df_p3.columns:
            target_t1   = df_p3[df_p3['freight_ratio'] > 0.2].copy()
            avg_freight = float(target_t1['freight_value'].mean()) \
                          if (not target_t1.empty and 'freight_value' in target_t1.columns) else 15.0
            run_simulation(
                target_df=target_t1,
                feature_changes={'freight_value': ('set', 0), 'freight_ratio': ('set', 0)},
                cost_per_head=avg_freight, tab_key="tab1",
                strategy_name="ส่งฟรี (Free Shipping)",
                rec_text=(f"ลูกค้าลังเลเพราะค่าส่งแพง (เฉลี่ย R$ {avg_freight:.0f})\n\n"
                          "👉 **Action:** ตั้ง `freight_value = 0` แล้วให้โมเดลทำนายซ้ำ")
            )
        else:
            st.error("ไม่พบข้อมูล freight_ratio")

    with tab2:
        st.subheader("💵 กลุ่มเสี่ยง Churn (Price Sensitivity)")
        disc_pct = st.radio("เลือก % ส่วนลด:", [10, 20], horizontal=True, key="disc_pct_t2")
        if 'price' in df_p3.columns:
            target_t2 = df_p3[df_p3['churn_probability'] > 0.5].copy()
            disc_cost = float(avg_ltv * disc_pct / 100)
            run_simulation(
                target_df=target_t2,
                feature_changes={
                    'price':         ('multiply', 1 - disc_pct/100),
                    'payment_value': ('multiply', 1 - disc_pct/100),
                },
                cost_per_head=disc_cost, tab_key="tab2",
                strategy_name=f"ส่วนลดสินค้า {disc_pct}%",
                rec_text=(f"ลด `price` ลง {disc_pct}% แล้วให้โมเดลทำนายซ้ำ\n\n"
                          f"👉 **Action:** เสนอ Coupon {disc_pct}% เฉพาะลูกค้า churn_prob > 50%")
            )
        else:
            st.error("ไม่พบข้อมูล price")

    with tab3:
        st.subheader("❤️ กลุ่มโดนเท / ของส่งช้า (Delay Recovery)")
        if 'delay_days' in df_p3.columns:
            target_t3 = df_p3[df_p3['delay_days'] > 0].copy()
            run_simulation(
                target_df=target_t3,
                feature_changes={
                    'delay_days':            ('set', 0),
                    'delivery_vs_estimated': ('clip_upper', 0),
                },
                cost_per_head=15.0, tab_key="tab3",
                strategy_name="SMS ขอโทษ + คูปองชดเชย",
                rec_text=("ตั้ง `delay_days = 0` (สมมติว่าปัญหาได้รับการแก้ไข)\n\n"
                          "👉 **Action:** ส่ง SMS ขอโทษทันที + แนบ Coupon ส่วนลดพิเศษ")
            )
        else:
            st.error("ไม่พบข้อมูล delay_days")

    with tab4:
        st.subheader("🛍️ กลุ่มซื้อหมวดเสี่ยง Churn สูง")
        if 'cat_churn_risk' in df_p3.columns:
            target_t4 = df_p3[df_p3['cat_churn_risk'] > 0.8].copy()
            run_simulation(
                target_df=target_t4,
                feature_changes={
                    'cat_churn_risk':        ('multiply', 0.6),
                    'payment_installments':  ('add', 2),
                },
                cost_per_head=10.0, tab_key="tab4",
                strategy_name="Cross-sell + ผ่อนได้นานขึ้น",
                rec_text=("ลด `cat_churn_risk` ลง 40% (จาก cross-sell หมวดซื้อซ้ำ)\n\n"
                          "👉 **Action:** ยิงแอดสินค้า Housewares + เพิ่ม installments")
            )
        else:
            st.error("ไม่พบข้อมูล cat_churn_risk")

# ==========================================
# PAGE 4: Logistics Insights
# ==========================================
elif page == "5. 🚛 Logistics Insights":
    import pydeck as pdk
    st.title("🚛 Logistics Insights")

    if 'customer_state' not in df.columns:
        st.error("❌ ไม่พบ customer_state"); st.stop()

    c1, c2 = st.columns(2)
    with c1:
        sel_c = st.multiselect("📦 หมวดสินค้า:", safe_cats(df), key="p4_cat")
    with c2:
        sel_s = st.multiselect("👥 สถานะ:",
            ['High Risk', 'Warning (Late > 1.5x)', 'Medium Risk', 'Lost (Late > 3x)', 'Active'],
            key="p4_status")

    df_log = df.copy()
    if sel_c: df_log = df_log[df_log['product_category_name'].isin(sel_c)]
    if sel_s: df_log = df_log[df_log['status'].isin(sel_s)]

    brazil = {
        'AC':[-9.02,-70.81],'AL':[-9.57,-36.78],'AM':[-3.41,-65.85],
        'AP':[0.90,-52.00], 'BA':[-12.58,-41.70],'CE':[-5.49,-39.32],
        'DF':[-15.79,-47.88],'ES':[-19.18,-40.30],'GO':[-15.82,-49.84],
        'MA':[-5.19,-45.16],'MG':[-19.81,-43.95],'MS':[-20.77,-54.78],
        'MT':[-12.96,-56.92],'PA':[-6.31,-52.46],'PB':[-7.24,-36.78],
        'PE':[-8.81,-36.95],'PI':[-7.71,-42.72],'PR':[-25.25,-52.02],
        'RJ':[-22.90,-43.17],'RN':[-5.40,-36.95],'RO':[-11.50,-63.58],
        'RR':[2.73,-62.07], 'RS':[-30.03,-51.22],'SC':[-27.24,-50.21],
        'SE':[-10.57,-37.38],'SP':[-23.55,-46.63],'TO':[-10.17,-48.33]
    }

    sm = df_log.groupby('customer_state').agg(
        payment_value=('payment_value', 'sum'),
        delivery_days=('delivery_days', 'mean'),
        delay_count=('delay_days', lambda x: (x > 0).sum()),
        churn_probability=('churn_probability', 'mean'),
        total_orders=('order_purchase_timestamp', 'count')
    ).reset_index()
    sm['lat'] = sm['customer_state'].map(lambda x: brazil.get(x, [0,0])[0])
    sm['lon'] = sm['customer_state'].map(lambda x: brazil.get(x, [0,0])[1])

    st.markdown("---")
    cs, k1, k2, k3 = st.columns([1.5, 1, 1, 1])
    with cs:
        zoom = st.selectbox("🔍 โฟกัสรัฐ:",
                            ["All"] + sorted(sm['customer_state'].unique()))
    disp     = sm if zoom == "All" else sm[sm['customer_state'] == zoom]
    view_lat = disp['lat'].mean() if zoom != "All" else -14.24
    view_lon = disp['lon'].mean() if zoom != "All" else -51.93
    view_z   = 6 if zoom != "All" else 3.5
    k1.metric("💰 ยอดเงิน",  f"R$ {disp['payment_value'].sum():,.0f}")
    k2.metric("🚚 ส่งเฉลี่ย", f"{disp['delivery_days'].mean():.1f} วัน")
    k3.metric("⚠️ ส่งช้า",   f"{int(disp['delay_count'].sum()):,} ครั้ง")

    cm_, ct_ = st.columns([2, 1])
    with cm_:
        st.subheader(f"🗺️ แผนที่ ({zoom})")
        sm['color'] = sm['churn_probability'].apply(
            lambda x: [231,76,60,200] if x>0.8 else
                      ([241,196,15,200] if x>0.5 else [46,204,113,200]))
        mx = sm['payment_value'].max()
        sm['radius'] = (sm['payment_value'] / mx * 400000) if mx > 0 else 10000
        layer = pdk.Layer("ScatterplotLayer", sm,
                          get_position='[lon,lat]', get_color='color', get_radius='radius',
                          pickable=True, opacity=0.8, stroked=True, filled=True,
                          radius_min_pixels=5, radius_max_pixels=60)
        tooltip = {"html": "<b>{customer_state}</b><br/>💰 R$ {payment_value}<br/>"
                           "🚚 {delivery_days} วัน<br/>⚠️ {delay_count} ครั้ง",
                   "style": {"backgroundColor": "steelblue", "color": "white"}}
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon,
                                             zoom=view_z, pitch=20),
            tooltip=tooltip, map_provider='carto', map_style='light'))

    with ct_:
        st.subheader("🚨 Top Issues")
        sort_m = st.radio("เรียงตาม:", ["ส่งช้า", "ความเสี่ยง"], horizontal=True, key="p4_sort")
        top_i = sm.sort_values('delay_count' if "ช้า" in sort_m else 'churn_probability',
                               ascending=False).head(10)
        st.dataframe(top_i[['customer_state', 'payment_value', 'delivery_days',
                             'delay_count', 'churn_probability']],
            column_config={
                "payment_value": st.column_config.NumberColumn("เงิน", format="R$%.0f"),
                "delivery_days": st.column_config.NumberColumn("ส่ง(วัน)", format="%.1f"),
                "churn_probability": st.column_config.ProgressColumn(
                    "Risk", format="%.2f", min_value=0, max_value=1)
            }, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("🏙️ เจาะลึกรายเมือง")
    if 'customer_city' in df_log.columns:
        city_m = df_log.groupby(['customer_state', 'customer_city']).agg(
            n=('customer_unique_id', 'count'),
            revenue=('payment_value', 'sum'),
            del_days=('delivery_days', 'mean'),
            late=('delay_days', lambda x: (x > 0).sum()),
            risk=('churn_probability', 'mean')
        ).reset_index()
        city_m = city_m[city_m['n'] >= 2]
        disp_c = city_m[city_m['customer_state'] == zoom] if zoom != "All" else city_m
        st.info(f"📍 {'รัฐ ' + zoom if zoom != 'All' else 'ทั่วประเทศ — Top 50 ที่ส่งช้ามากสุด'}")
        st.dataframe(disp_c.sort_values('late', ascending=False).head(50),
            column_config={
                "n": st.column_config.NumberColumn("ลูกค้า"),
                "revenue": st.column_config.NumberColumn("ยอดเงิน", format="R$%.0f"),
                "del_days": st.column_config.NumberColumn("ส่ง(วัน)", format="%.1f"),
                "late": st.column_config.NumberColumn("ส่งช้า"),
                "risk": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1)
            }, hide_index=True, use_container_width=True)

# ==========================================
# PAGE 5: Seller Audit
# ==========================================
elif page == "6. 🏪 Seller Audit":
    st.title("🏪 Seller Audit")

    if 'seller_id' not in df.columns:
        st.error("❌ ไม่พบ seller_id"); st.stop()

    c1, c2 = st.columns(2)
    with c1:
        sel_c = st.multiselect("📦 หมวดสินค้า:", safe_cats(df), key="p5c")
    with c2:
        sel_s = st.multiselect("👥 สถานะ:",
            ['High Risk', 'Warning (Late > 1.5x)', 'Medium Risk', 'Lost (Late > 3x)', 'Active'],
            key="p5s")

    dfs = df.copy()
    if sel_c: dfs = dfs[dfs['product_category_name'].isin(sel_c)]
    if sel_s: dfs = dfs[dfs['status'].isin(sel_s)]

    agg = {'order_purchase_timestamp': 'count', 'payment_value': 'sum',
           'churn_probability': 'mean', 'delivery_days': 'mean'}
    if 'review_score' in dfs.columns: agg['review_score'] = 'mean'
    ss = dfs.groupby('seller_id').agg(agg).reset_index()
    ss = ss.rename(columns={'order_purchase_timestamp': 'orders'})
    if 'review_score' not in ss.columns: ss['review_score'] = np.nan
    ss = ss[ss['orders'] >= 3]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🏪 ร้านค้า",     f"{len(ss):,}")
    k2.metric("💸 ยอดขายรวม",   f"R$ {ss['payment_value'].sum():,.0f}")
    k3.metric("⭐ รีวิวเฉลี่ย", f"{ss['review_score'].mean():.2f}"
              if ss['review_score'].notna().any() else "N/A")
    k4.metric("🚚 ส่งเฉลี่ย",   f"{ss['delivery_days'].mean():.1f} วัน")

    st.markdown("---")
    cs_, cd_ = st.columns([1, 3])
    with cs_:
        sort_m = st.radio("เรียงตาม:", [
            "🚨 ความเสี่ยง", "🐢 ส่งช้า", "⭐ คะแนนต่ำ", "💸 ยอดขาย", "📦 ปริมาณ"])
    with cd_:
        if "ความเสี่ยง" in sort_m:  sdf = ss.sort_values('churn_probability', ascending=False)
        elif "ส่งช้า" in sort_m:    sdf = ss.sort_values('delivery_days', ascending=False)
        elif "คะแนนต่ำ" in sort_m:  sdf = ss.sort_values('review_score', ascending=True)
        elif "ยอดขาย" in sort_m:    sdf = ss.sort_values('payment_value', ascending=False)
        else:                        sdf = ss.sort_values('orders', ascending=False)
        st.dataframe(sdf, column_config={
            "orders": st.column_config.NumberColumn("Orders"),
            "payment_value": st.column_config.NumberColumn("Revenue", format="R$%.0f"),
            "delivery_days": st.column_config.NumberColumn("ส่ง(วัน)", format="%.1f"),
            "review_score": st.column_config.NumberColumn("Review", format="%.1f⭐"),
            "churn_probability": st.column_config.ProgressColumn(
                "Risk", format="%.2f", min_value=0, max_value=1)
        }, hide_index=True, use_container_width=True, height=600)

# ==========================================
# PAGE 6: Buying Cycle
# ==========================================
elif page == "4. 🔄 Buying Cycle Analysis":
    st.title("🔄 Buying Cycle Analysis")

    sel_c  = st.multiselect("📦 หมวดสินค้า:", safe_cats(df), key="p6c")
    df_cy  = df[df['product_category_name'].isin(sel_c)].copy() if sel_c else df.copy()

    g_avg  = df['cat_median_days'].mean()
    c_avg  = df_cy['cat_median_days'].mean()
    c_late = df_cy['lateness_score'].mean() if 'lateness_score' in df_cy.columns else 0
    fast   = (df_cy['cat_median_days'] <= 30).sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("⏱️ รอบซื้อเฉลี่ย", f"{c_avg:.0f} วัน",
              f"{c_avg-g_avg:+.0f} วัน vs ภาพรวม", delta_color="inverse")
    m2.metric("🐢 ความล่าช้า", f"{c_late:.2f}x")
    m3.metric("📅 ซื้อซ้ำใน 30 วัน", f"{fast:,} คน")

    st.markdown("---")
    st.subheader("📈 Buying Cycle Trend")
    if 'order_purchase_timestamp' in df_cy.columns:
        tmp2 = df_cy.sort_values(['customer_unique_id', 'order_purchase_timestamp'])
        tmp2['prev_t'] = tmp2.groupby('customer_unique_id')['order_purchase_timestamp'].shift(1)
        tmp2['gap']    = (tmp2['order_purchase_timestamp'] - tmp2['prev_t']).dt.days
        rep = tmp2[tmp2['gap'].notna() & (tmp2['gap'] > 0)].copy()
        if not rep.empty:
            rep['month_year'] = rep['order_purchase_timestamp'].dt.to_period('M')
            tgap = rep.groupby('month_year')['gap'].mean().reset_index()
            if len(tgap) > 1:
                tgap = tgap.iloc[:-1]
                tgap['Date'] = pd.to_datetime(tgap['month_year'].astype(str))
                st.altair_chart(
                    alt.Chart(tgap).mark_line(point=True, strokeWidth=3).encode(
                        x=alt.X('Date', axis=alt.Axis(format='%b %Y')),
                        y=alt.Y('gap', title='ระยะเวลาซื้อซ้ำเฉลี่ย (วัน)',
                                scale=alt.Scale(zero=False)),
                        color=alt.value('#e67e22'),
                        tooltip=['Date', alt.Tooltip('gap', format='.1f', title='วัน')]
                    ).properties(height=350), use_container_width=True)
            else:
                st.info("ข้อมูลไม่เพียงพอสำหรับ Trend")
        else:
            st.info("ไม่พบลูกค้าที่ซื้อซ้ำ")

    st.markdown("---")
    st.subheader("📋 รายละเอียดรายหมวด")
    summ = df_cy.groupby('product_category_name').agg(
        Customers=('customer_unique_id', 'count'),
        Cycle_Days=('cat_median_days', 'mean'),
        Late_Score=('lateness_score', 'mean'),
        Churn_Risk=('churn_probability', 'mean')
    ).reset_index().sort_values('Cycle_Days')
    st.dataframe(summ, column_config={
        "Customers":  st.column_config.NumberColumn("ลูกค้า", format="%d คน"),
        "Cycle_Days": st.column_config.NumberColumn("รอบซื้อ", format="%.0f วัน"),
        "Late_Score": st.column_config.NumberColumn("ความล่าช้า", format="%.2fx"),
        "Churn_Risk": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1)
    }, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("📅 Seasonal Heatmap")
    if 'order_purchase_timestamp' in df_cy.columns:
        sea = df_cy.copy()
        sea['month_num']  = sea['order_purchase_timestamp'].dt.month
        sea['month_name'] = sea['order_purchase_timestamp'].dt.strftime('%b')
        hm = sea.groupby(['product_category_name', 'month_num', 'month_name']
                         ).size().reset_index(name='vol')
        top_c = sea['product_category_name'].value_counts().head(15).index.tolist()
        hm    = hm[hm['product_category_name'].isin(top_c)]
        if not hm.empty:
            st.altair_chart(
                alt.Chart(hm).mark_rect().encode(
                    x=alt.X('month_name', sort=['Jan','Feb','Mar','Apr','May','Jun',
                                                'Jul','Aug','Sep','Oct','Nov','Dec'],
                            title='เดือน'),
                    y=alt.Y('product_category_name', title='หมวด'),
                    color=alt.Color('vol', scale=alt.Scale(scheme='orangered'), title='ยอดขาย'),
                    tooltip=['product_category_name', 'month_name',
                             alt.Tooltip('vol', format=',')]
                ).properties(height=500), use_container_width=True)
            st.info("💡 สีส้มเข้ม = High Season → เตรียมสต็อกล่วงหน้า")

# ==========================================
# PAGE 7: Customer Detail
# ==========================================
elif page == "7. 🔍 Customer Detail":
    st.title("🔍 Customer Deep Dive")

    with st.expander("🔎 Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_status = st.multiselect("สถานะ:",
                ['High Risk', 'Warning (Late > 1.5x)', 'Medium Risk', 'Lost (Late > 3x)', 'Active'],
                default=['High Risk', 'Warning (Late > 1.5x)'])
        with f2:
            sel_cats = st.multiselect("หมวดสินค้า:", safe_cats(df))
        with f3:
            search_id = st.text_input("ค้นหา Customer ID:", "")

    mask = df['status'].isin(sel_status)
    if sel_cats:  mask = mask & df['product_category_name'].isin(sel_cats)
    if search_id: mask = mask & df['customer_unique_id'].str.contains(
                              search_id, case=False, na=False)
    filtered = df[mask]

    if 'product_category_name' in df.columns and not filtered.empty:
        cat_ov   = df.groupby('product_category_name').agg(
            Total=('customer_unique_id', 'count'),
            Cycle=('cat_median_days', 'mean')).reset_index()
        cat_risk = filtered.groupby('product_category_name').agg(
            Risk=('customer_unique_id', 'count')).reset_index()
        cat_s    = cat_risk.merge(cat_ov, on='product_category_name', how='left')
        cat_s['Risk_Pct'] = cat_s['Risk'] / cat_s['Total']
        cat_s = cat_s.sort_values('Risk', ascending=False)

        cc, ct = st.columns([1.5, 2.5])
        with cc:
            st.subheader("📊 Top 10 หมวดเสี่ยง")
            base   = alt.Chart(cat_s.head(10)).encode(
                y=alt.Y('product_category_name', sort='-x', title=None))
            b_tot  = base.mark_bar(color='#f0f2f6').encode(x='Total')
            b_risk = base.mark_bar(color='#e74c3c').encode(x='Risk')
            st.altair_chart(b_tot + b_risk, use_container_width=True)
        with ct:
            st.subheader("📋 รายละเอียด")
            st.dataframe(cat_s, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(f"📄 รายชื่อลูกค้า ({len(filtered):,} คน)")
    show = [c for c in ['customer_unique_id', 'status', 'churn_probability',
                        'lateness_score', 'cat_median_days', 'payment_value',
                        'product_category_name'] if c in df.columns]
    st.dataframe(
        filtered[show].sort_values('churn_probability', ascending=False),
        column_config={
            "churn_probability": st.column_config.ProgressColumn(
                "Risk", format="%.2f", min_value=0, max_value=1),
            "lateness_score": st.column_config.NumberColumn("Late", format="%.1fx")
        }, use_container_width=True)
