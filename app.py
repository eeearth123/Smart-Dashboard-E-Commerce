import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt

# --- 1. ตั้งค่า Page Config (ต้องอยู่บรรทัดแรกสุด) ---
st.set_page_config(
    page_title="Olist Business Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ฟังก์ชันโหลดโมเดล ---
@st.cache_resource
def load_model_objects():
    try:
        model = joblib.load('olist_churn_rf_model.pkl')
        features = joblib.load('model_features.pkl')
        return model, features
    except FileNotFoundError:
        return None, None

loaded_model, model_features = load_model_objects()

# --- 3. สร้างข้อมูลจำลอง (Mock Data) เพื่อโชว์กราฟในหน้า Dashboard ---
# (ในงานจริง ส่วนนี้จะเปลี่ยนเป็นการโหลดไฟล์ csv ทั้งหมดของร้านมาแสดง)
@st.cache_data
def get_mock_dashboard_data():
    np.random.seed(42)
    data_size = 200
    df = pd.DataFrame({
        'customer_id': [f'CUST-{i:04d}' for i in range(data_size)],
        'delivery_days': np.random.normal(15, 5, data_size),
        'review_score': np.random.choice([1, 2, 3, 4, 5], data_size, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
        'total_spend': np.random.exponential(500, data_size),
        'churn_prob': np.random.uniform(0, 1, data_size)
    })
    # กำหนดสถานะ Churn ตามความน่าจะเป็น (สมมติ)
    df['status'] = df['churn_prob'].apply(lambda x: 'High Risk (Churn)' if x > 0.6 else 'Safe (Active)')
    return df

dashboard_data = get_mock_dashboard_data()

# --- 4. Sidebar Navigation (เมนูเลือกหน้า) ---
st.sidebar.title("🛍️ Olist Analytics")
page = st.sidebar.radio("เลือกเมนูใช้งาน", ["📊 ภาพรวมธุรกิจ (Overview)", "🔍 ตรวจสอบรายบุคคล (Predictor)", "🧠 เจาะลึกผลโมเดล (Model Insights)"])
st.sidebar.markdown("---")
st.sidebar.info(f"Model Status: {'✅ Ready' if loaded_model else '❌ Not Found'}")

# ==============================================================================
# PAGE 1: 📊 ภาพรวมธุรกิจ (Overview)
# ==============================================================================
if page == "📊 ภาพรวมธุรกิจ (Overview)":
    st.title("📊 Business Health Overview")
    st.markdown("สรุปภาพรวมความเสี่ยงลูกค้าในระบบ (จำลองข้อมูล)")

    # 1. Top Metrics (KPIs)
    col1, col2, col3, col4 = st.columns(4)
    high_risk_count = len(dashboard_data[dashboard_data['status'] == 'High Risk (Churn)'])
    avg_score = dashboard_data['review_score'].mean()
    
    col1.metric("ลูกค้าทั้งหมด", f"{len(dashboard_data)} คน")
    col2.metric("ลูกค้าเสี่ยงหลุด (High Risk)", f"{high_risk_count} คน", delta="-ระวัง", delta_color="inverse")
    col3.metric("คะแนนรีวิวเฉลี่ย", f"{avg_score:.2f} / 5.0")
    col4.metric("Retention Rate (ประมาณการ)", f"{(1 - high_risk_count/len(dashboard_data))*100:.1f}%")

    st.markdown("---")

    # 2. Charts Layout
    c1, c2 = st.columns((2, 1))

    with c1:
        st.subheader("📦 ความสัมพันธ์: เวลาส่งของ vs โอกาส Churn")
        # กราฟ Scatter Plot
        chart = alt.Chart(dashboard_data).mark_circle(size=60).encode(
            x=alt.X('delivery_days', title='จำนวนวันรอของ (Days)'),
            y=alt.Y('churn_prob', title='โอกาส Churn (%)'),
            color=alt.Color('status', legend=alt.Legend(title="Status")),
            tooltip=['customer_id', 'delivery_days', 'review_score']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
        st.caption("*Insight: ยิ่งรอนาน (ไปทางขวา) จุดสีแดงยิ่งอยู่สูง แปลว่าโอกาสหนียิ่งมาก")

    with c2:
        st.subheader("⭐ สัดส่วนความเสี่ยงตามเกรดรีวิว")
        # กราฟแท่ง
        bar_chart = alt.Chart(dashboard_data).mark_bar().encode(
            x=alt.X('review_score:O', title='คะแนนรีวิว'),
            y='count()',
            color='status'
        )
        st.altair_chart(bar_chart, use_container_width=True)
        st.caption("*Insight: คะแนนรีวิวต่ำ (1-2 ดาว) มีสัดส่วนลูกค้าเสี่ยงสูงมาก")

    # 3. Table of Urgent Action
    st.subheader("🚨 รายชื่อลูกค้าที่ต้องรีบดูแล (Top 5 Highest Risk)")
    urgent_customers = dashboard_data.sort_values('churn_prob', ascending=False).head(5)
    st.dataframe(urgent_customers[['customer_id', 'churn_prob', 'total_spend', 'review_score']], use_container_width=True)


# ==============================================================================
# PAGE 2: 🔍 ตรวจสอบรายบุคคล (Predictor) - อันเดิมที่คุณทำไว้
# ==============================================================================
elif page == "🔍 ตรวจสอบรายบุคคล (Predictor)":
    st.title("🔍 Individual Customer Check")
    st.markdown("เครื่องมือประเมินความเสี่ยงลูกค้าทีละราย สำหรับทีม Customer Service")

    if loaded_model is None:
        st.error("ไม่พบไฟล์โมเดล! กรุณาเช็คว่าไฟล์ .pkl อยู่ในโฟลเดอร์เดียวกับ app.py")
    else:
        # Layout: แบ่งซ้ายขวา Input | Output
        col_input, col_result = st.columns([1, 1.5])

        with col_input:
            st.subheader("📝 กรอกข้อมูลลูกค้า")
            # Input Fields
            delivery_days_mean = st.number_input("รอของเฉลี่ย (วัน)", value=15.0)
            delay_days_mean = st.number_input("ส่งช้าเฉลี่ย (วัน)", value=2.0)
            avg_review_score = st.slider("คะแนนรีวิวเฉลี่ย", 1.0, 5.0, 4.5)
            total_late_orders = st.number_input("จำนวนออเดอร์ที่ส่งช้า", value=1)
            
            with st.expander("ข้อมูลเพิ่มเติม (Optional)"):
                freight_value_mean = st.number_input("ค่าส่งเฉลี่ย", value=35.5)
                is_high_freight = st.selectbox("กลุ่มค่าส่งแพง?", [0, 1], index=1)
                monetary_value = st.number_input("ยอดซื้อรวม", value=500.0)
                # ค่า Default อื่นๆ ที่จำเป็นต่อโมเดล
                delivery_days_max = 20.0
                shipping_cost_per_gram = 0.05
                product_weight_g_mean = 500.0
                freight_ratio_mean = 0.15
                product_photos_qty_mean = 2.0
                min_review_score = 4
                avg_basket_value = 120.0
                price_mean = 100.0
                frequency = 1
                category_diversity = 1
                is_shipping_ripoff = 0

        with col_result:
            st.subheader("🔮 ผลการวิเคราะห์")
            # ปุ่มกดทำนาย
            if st.button("ประเมินความเสี่ยง (Analyze)", use_container_width=True, type="primary"):
                # เตรียม Data
                input_data = pd.DataFrame([{
                    'freight_value_mean': freight_value_mean,
                    'delivery_days_mean': delivery_days_mean,
                    'delay_days_mean': delay_days_mean,
                    'delivery_days_max': delivery_days_max,
                    'shipping_cost_per_gram': shipping_cost_per_gram,
                    'product_weight_g_mean': product_weight_g_mean,
                    'freight_ratio_mean': freight_ratio_mean,
                    'avg_basket_value': avg_basket_value,
                    'is_high_freight_customer': is_high_freight,
                    'price_mean': price_mean,
                    'monetary_value': monetary_value,
                    'product_photos_qty_mean': product_photos_qty_mean,
                    'total_late_orders': total_late_orders,
                    'min_review_score': min_review_score,
                    'avg_review_score': avg_review_score,
                    'frequency': frequency,
                    'category_diversity': category_diversity,
                    'is_shipping_ripoff': is_shipping_ripoff
                }])

                try:
                    input_data = input_data[model_features]
                    prediction = loaded_model.predict(input_data)[0]
                    prob = loaded_model.predict_proba(input_data)[0][1] * 100
                    
                    # Card แสดงผล
                    st.markdown("---")
                    if prob > 50:
                        st.error(f"⚠️ **HIGH RISK: มีความเสี่ยงสูง ({prob:.2f}%)**")
                        st.write("ลูกค้ามีแนวโน้มจะ **Churn** (เลิกใช้บริการ)")
                        st.progress(int(prob), text="Risk Level")
                        st.warning("💡 **คำแนะนำ:** ควรเสนอคูปองส่วนลดหรือโทรสอบถามความพึงพอใจด่วน")
                    else:
                        st.success(f"✅ **LOW RISK: ลูกค้าปกติ ({prob:.2f}%)**")
                        st.write("ลูกค้ามีแนวโน้มจะ **Stay** (อยู่ต่อ)")
                        st.progress(int(prob), text="Risk Level")
                
                except Exception as e:
                    st.error(f"Error: {e}")

# ==============================================================================
# PAGE 3: 🧠 เจาะลึกผลโมเดล (Model Insights)
# ==============================================================================
elif page == "🧠 เจาะลึกผลโมเดล (Model Insights)":
    st.title("🧠 Model Insights & Explanation")
    st.markdown("หน้านี้ช่วยให้ผู้บริหารเข้าใจว่า **ปัจจัยอะไรที่มีผลต่อลูกค้ามากที่สุด**")

    if loaded_model:
        # ดึง Feature Importance จาก Random Forest
        importances = loaded_model.feature_importances_
        feature_names = model_features
        
        # สร้าง DataFrame
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(10) # เอาแค่ Top 10
        
        # กราฟแท่งแนวนอน
        st.subheader("🏆 Top 10 ปัจจัยที่ทำให้ลูกค้าหนี (Churn Drivers)")
        
        chart_fi = alt.Chart(fi_df).mark_bar(color='#FF4B4B').encode(
            x=alt.X('Importance', title='ระดับความสำคัญ (Importance Score)'),
            y=alt.Y('Feature', sort='-x', title='ตัวแปร (Feature)'),
            tooltip=['Feature', 'Importance']
        )
        st.altair_chart(chart_fi, use_container_width=True)
        
        st.info("""
        **วิธีการอ่านค่า:**
        * แท่งยิ่งยาว แปลว่าปัจจัยนั้นมีผลต่อการตัดสินใจของลูกค้ามากที่สุด
        * โดยปกติ **Delivery Days (วันส่ง)** และ **Review Score (คะแนนรีวิว)** มักจะติดอันดับต้นๆ ใน Olist
        """)
        
    else:
        st.warning("ไม่พบโมเดล ไม่สามารถแสดง Insights ได้")

# ... (ต่อจากโค้ดกราฟแท่ง Feature Importance เดิม) ...
st.markdown("---")
st.subheader("🌳 Visualization: โครงสร้างต้นไม้ตัดสินใจ (Decision Tree)")
st.write("เนื่องจาก Random Forest ประกอบด้วยต้นไม้ 100 ต้น เราจึงขอยกตัวอย่าง **ต้นไม้ต้นที่ 1** มาแสดงให้ดูโครงสร้างการคิดครับ")

# ปุ่มกดเพื่อโชว์ (เพราะมันโหลดหนัก จะได้ไม่หน่วงถ้าไม่กด)
if st.button("กดเพื่อแสดงแผนภาพต้นไม้ (Tree Diagram)"):
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree
    # ดึงต้นไม้ต้นแรกออกมา (Estimator 0)
    estimator = loaded_model.estimators_[0]
    # ตั้งค่ารูปภาพ (figsize ต้องใหญ่หน่อย ไม่งั้นมองไม่เห็น)
    fig, ax = plt.subplots(figsize=(20, 10))
    # วาดกราฟ
    plot_tree(estimator, 
              feature_names=model_features,
              class_names=['Stay', 'Churn'],
              filled=True, 
              rounded=True,
              fontsize=10,
              max_depth=3,  # <--- สำคัญ: ปรับเลขนี้เพื่อดูความลึก (ถ้าใส่ 10 จะดูยากมาก แนะนำ 3 ก่อน)
              ax=ax)
    # แสดงผลใน Streamlit
    st.pyplot(fig)
    st.caption("หมายเหตุ: แสดงความลึกแค่ 3 ชั้นแรกเพื่อให้ดูง่าย (โมเดลจริงลึก 10 ชั้น)")


