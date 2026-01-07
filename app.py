import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Olist Churn Dashboard", page_icon="📊", layout="wide")

st.title("📊 Olist Customer Analytics & Churn Prediction")
st.markdown("### แดชบอร์ดวิเคราะห์แนวโน้มลูกค้า (Demo Version)")

# --- 2. ฟังก์ชันโหลดโมเดล (ใช้ชื่อไฟล์เดิม) ---
@st.cache_resource
def load_model_objects():
    try:
        model = joblib.load('olist_churn_rf_model.pkl')
        features = joblib.load('model_features.pkl')
        return model, features
    except FileNotFoundError:
        return None, None

loaded_model, model_features = load_model_objects()

# --- 3. Sidebar: รับข้อมูล (เหมือนเดิม แต่จัดกลุ่มให้ดูง่ายขึ้น) ---
with st.sidebar:
    st.header("📝 ข้อมูลลูกค้า")
    
    with st.expander("🚚 ข้อมูลการขนส่ง (Delivery)", expanded=True):
        delivery_days_mean = st.slider("รอของเฉลี่ย (วัน)", 1, 60, 15)
        delay_days_mean = st.slider("ส่งช้าเฉลี่ย (วัน)", 0, 30, 2)
        freight_value_mean = st.number_input("ค่าส่งเฉลี่ย (BRL)", value=35.5)
        is_high_freight_customer = st.selectbox("เป็นลูกค้ากลุ่มค่าส่งแพง?", [0, 1], index=1)

    with st.expander("💰 ข้อมูลการใช้จ่าย (Spending)", expanded=True):
        monetary_value = st.number_input("ยอดซื้อสะสมรวม (BRL)", value=500.0)
        avg_basket_value = st.number_input("ยอดต่อตะกร้าเฉลี่ย (BRL)", value=120.0)
        price_mean = st.number_input("ราคาสินค้าเฉลี่ย", value=100.0)
        
    with st.expander("⭐ พฤติกรรมอื่นๆ (Behavior)"):
        total_late_orders = st.number_input("จำนวนออเดอร์ที่ส่งช้า", value=1)
        avg_review_score = st.slider("คะแนนรีวิวเฉลี่ย", 1.0, 5.0, 4.5)
        # ตัวแปรอื่นๆ ที่จำเป็น (ใส่ค่า Default ไว้ก่อน)
        delivery_days_max = 20.0
        shipping_cost_per_gram = 0.05
        product_weight_g_mean = 500.0
        freight_ratio_mean = 0.15
        product_photos_qty_mean = 2.0
        min_review_score = 4

# --- 4. เตรียมข้อมูลสำหรับกราฟและโมเดล ---
# สร้าง DataFrame
input_data = pd.DataFrame([{
    'freight_value_mean': freight_value_mean,
    'delivery_days_mean': delivery_days_mean,
    'delay_days_mean': delay_days_mean,
    'delivery_days_max': delivery_days_max,
    'shipping_cost_per_gram': shipping_cost_per_gram,
    'product_weight_g_mean': product_weight_g_mean,
    'freight_ratio_mean': freight_ratio_mean,
    'avg_basket_value': avg_basket_value,
    'is_high_freight_customer': is_high_freight_customer,
    'price_mean': price_mean,
    'monetary_value': monetary_value,
    'product_photos_qty_mean': product_photos_qty_mean,
    'total_late_orders': total_late_orders,
    'min_review_score': min_review_score,
    'avg_review_score': avg_review_score,
    # --- แก้ Error: ใส่ค่า Default ให้ตัวที่ขาดไป ---
    'frequency': 1,             # สมมติว่าซื้อ 1 ครั้ง
    'category_diversity': 1,    # สมมติว่าซื้อหมวดเดียว
    'is_shipping_ripoff': 0     # สมมติว่าค่าส่งไม่โหดร้าย
}])

# --- 5. แสดงผลกราฟ (Visualizations) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Customer DNA Profile")
    # เตรียมข้อมูลทำกราฟ (เลือกเฉพาะค่าสำคัญๆ มาโชว์)
    chart_data = pd.DataFrame({
        'Factor': ['Delivery Days', 'Review Score (x10)', 'Late Orders (x5)', 'Delay Days'],
        'Value': [delivery_days_mean, avg_review_score*10, total_late_orders*5, delay_days_mean]
    })
    
    # แสดง Bar Chart
    st.bar_chart(chart_data.set_index('Factor'))
    st.caption("*กราฟแสดงปัจจัยเสี่ยง: แท่งยิ่งสูง ยิ่งมีผลต่อความรู้สึกลูกค้า")

with col2:
    st.subheader("🔮 ผลการทำนาย")
    
    if loaded_model is not None:
        try:
            # จัดเรียงคอลัมน์ให้ตรงเป๊ะๆ
            input_data = input_data[model_features]
            
            # ทำนาย
            prob = loaded_model.predict_proba(input_data)[0][1] # โอกาส Churn (0-1)
            
            # แสดงเป็น Metric สวยๆ
            churn_percentage = prob * 100
            
            if churn_percentage > 50:
                st.error("มีความเสี่ยงสูง (High Risk)")
                st.metric(label="โอกาสเลิกซื้อ (Churn Probability)", value=f"{churn_percentage:.2f}%", delta="เสี่ยง", delta_color="inverse")
                st.progress(int(churn_percentage), text="Risk Level")
            else:
                st.success("ลูกค้าปกติ (Low Risk)")
                st.metric(label="โอกาสเลิกซื้อ (Churn Probability)", value=f"{churn_percentage:.2f}%", delta="ปลอดภัย")
                st.progress(int(churn_percentage), text="Risk Level")
                
        except Exception as e:
            st.warning(f"ยังทำนายไม่ได้สมบูรณ์ แต่กราฟทำงานได้ปกติครับ ({e})")
    else:
        st.info("โหมดแสดงผลกราฟ (ยังไม่พบโมเดล)")

st.markdown("---")
st.write("📌 **Note:** ระบบใส่ค่า Default ให้ตัวแปร `frequency`, `category_diversity` อัตโนมัติเพื่อให้รันผ่านครับ")
