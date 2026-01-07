import streamlit as st
import pandas as pd
import joblib

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Olist Churn Prediction", page_icon="📦")

st.title("📦 Olist Churn Prediction AI")
st.write("ระบบทำนายแนวโน้มลูกค้าว่าจะเลิกใช้บริการ (Churn) หรือไม่")

# --- 2. ฟังก์ชันโหลดโมเดล (ใช้ชื่อไฟล์ตามที่คุณส่งมา) ---
@st.cache_resource
def load_model_objects():
    try:
        # โหลดไฟล์โมเดลและชื่อฟีเจอร์
        model = joblib.load('olist_churn_rf_model')
        features = joblib.load('model_features')
        return model, features
    except FileNotFoundError as e:
        return None, None

# เรียกใช้ฟังก์ชันโหลด
loaded_model, model_features = load_model_objects()

# --- 3. ส่วนแสดงผลหลัก ---
if loaded_model is None or model_features is None:
    st.error("❌ ไม่พบไฟล์โมเดล! กรุณาเช็คว่าไฟล์ 'olist_churn_rf_model.pkl' และ 'model_features.pkl' วางอยู่ที่เดียวกับ app.py หรือยัง?")
else:
    # สร้าง Sidebar สำหรับกรอกข้อมูล (จำลองคอลัมน์ตามตัวอย่างที่คุณให้มา)
    st.sidebar.header("📝 กรอกข้อมูลลูกค้า")
    
    # รับค่าต่างๆ (ตั้งค่าเริ่มต้นตามตัวอย่าง sample_input ของคุณ)
    freight_value_mean = st.sidebar.number_input("ค่าส่งเฉลี่ย (Freight Value)", value=35.5)
    delivery_days_mean = st.sidebar.number_input("รอของกี่วัน (Avg Delivery Days)", value=15.0)
    delay_days_mean = st.sidebar.number_input("ส่งช้าเฉลี่ยกี่วัน (Avg Delay Days)", value=2.0)
    delivery_days_max = st.sidebar.number_input("รอนานสุดกี่วัน (Max Delivery Days)", value=20.0)
    
    # กลุ่มข้อมูลสินค้าและราคา
    shipping_cost_per_gram = st.sidebar.number_input("ค่าส่งต่อกรัม", value=0.05, format="%.4f")
    product_weight_g_mean = st.sidebar.number_input("น้ำหนักสินค้าเฉลี่ย (g)", value=500.0)
    freight_ratio_mean = st.sidebar.number_input("สัดส่วนค่าส่ง (Freight Ratio)", value=0.15)
    avg_basket_value = st.sidebar.number_input("ยอดซื้อต่อตะกร้าเฉลี่ย", value=120.0)
    
    # ข้อมูลพฤติกรรม
    is_high_freight_customer = st.sidebar.selectbox("เป็นลูกค้าค่าส่งแพงหรือไม่?", options=[0, 1], index=1)
    price_mean = st.sidebar.number_input("ราคาสินค้าเฉลี่ย", value=100.0)
    monetary_value = st.sidebar.number_input("มูลค่ารวม (Monetary)", value=500.0)
    product_photos_qty_mean = st.sidebar.number_input("จำนวนรูปสินค้าเฉลี่ย", value=2.0)
    
    # ข้อมูลรีวิวและการส่งช้า
    total_late_orders = st.sidebar.number_input("จำนวนออเดอร์ที่ส่งช้า", value=1)
    min_review_score = st.sidebar.slider("คะแนนรีวิวต่ำสุด", 1, 5, 4)
    avg_review_score = st.sidebar.slider("คะแนนรีวิวเฉลี่ย", 1.0, 5.0, 4.5)

    # --- 4. ปุ่มกดทำนาย ---
    if st.button("🔮 ทำนายผล (Predict)"):
        # รวบรวมข้อมูลเข้า DataFrame
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
            'avg_review_score': avg_review_score
        }])

        try:
            # จัดเรียงคอลัมน์ให้ตรงกับตอนเทรนเป๊ะๆ
            input_data = input_data[model_features]
            
            # ทำนายผล
            prediction = loaded_model.predict(input_data)[0]
            prob = loaded_model.predict_proba(input_data)[0][1] # โอกาสเกิด Class 1 (Churn)

            # แสดงผลลัพธ์
            st.markdown("---")
            if prediction == 1:
                st.error(f"⚠️ **ผลทำนาย: ลูกค้ามีแนวโน้มจะหนี (Churn)**")
                st.write(f"ความมั่นใจของโมเดล: **{prob*100:.2f}%**")
            else:
                st.success(f"✅ **ผลทำนาย: ลูกค้ายังอยู่ (Stay)**")
                st.write(f"ความเสี่ยงที่จะหนีเพียง: **{prob*100:.2f}%**")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")

            st.info("คำแนะนำ: ลองเช็คว่าไฟล์ model_features.pkl ตรงกับเวอร์ชันล่าสุดที่เทรนมาหรือไม่")
