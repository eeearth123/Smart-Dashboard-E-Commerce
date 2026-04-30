# 📊 Dashboard Guide — Olist Churn Intelligence Platform
 
> **Streamlit Cloud** · **Google BigQuery** · **LightGBM AI Model**  
> ระบบวิเคราะห์ลูกค้าเสี่ยง Churn แบบ Real-time พร้อมจำลอง ROI แคมเปญก่อนใช้งบจริง
 
---
 
## 📋 สารบัญ
 
| หน้า | ชื่อ | จุดประสงค์หลัก |
|------|------|----------------|
| [1](#1--business-overview) | 💰 Business Overview | ภาพรวมรายได้และหมวดสินค้า |
| [2](#2--churn-overview) | 📊 Churn Overview | ภาพรวมความเสี่ยง Churn ทั้งบริษัท |
| [3](#3--action-plan--roi-simulator) | 🎯 Action Plan | จำลองแคมเปญและคำนวณ ROI |
| [4](#4--buying-cycle-analysis) | 🔄 Buying Cycle | วิเคราะห์รอบการซื้อซ้ำต่อหมวดสินค้า |
| [5](#5--logistics-insights) | 🚛 Logistics | แผนที่วิเคราะห์การส่งของรายรัฐ |
| [6](#6--seller-audit) | 🏪 Seller Audit | จัดอันดับร้านค้าตามความเสี่ยง |
| [7](#7--customer-detail) | 🔍 Customer Detail | ค้นหาลูกค้ารายคน |
 
---
 
## 1 · 💰 Business Overview
 
### 📐 Layout Guide — แต่ละส่วนของหน้านี้คืออะไร

![Business Overview Layout](./Docs/images/p1annotated.png)

### 📸 Dashboard — ผลลัพธ์จริง
 
> `![Churn Overview Dashboard](docs/images/p2_dashboard.png)`

---
 
### 💡 Insights ที่ได้จากหน้านี้
 
| Insight | ประโยชน์ต่อธุรกิจ |
|---------|-----------------|
| **MoM Growth** แสดงเดือนที่รายได้ตก | ระบุช่วงที่ควรเร่งแคมเปญ ก่อนยอดหล่น |
| **สีแท่งหมวดสินค้า** บอก Churn Risk | รู้ทันทีว่าหมวดไหนทำเงินมากแต่ลูกค้าหนีด้วย ต้องระวังเป็นพิเศษ |
| **Avg Order Value** ต่อหมวด | วางราคา bundle หรือ upsell ได้ตรงกลุ่ม |
 
> **ตัวอย่าง insight จากข้อมูล Olist:**  
> หมวด `Health & Beauty` มียอดขายสูงเป็นอันดับต้น แต่ Churn Risk สูงกว่าค่าเฉลี่ย  
> → ควรออก loyalty program เฉพาะหมวดนี้
 
---
 
---
