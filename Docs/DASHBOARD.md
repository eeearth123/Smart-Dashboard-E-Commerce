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
| **Trend 2 เส้น** เปรียบเทียบ Rule vs AI | เห็นว่า AI จับสัญญาณล่วงหน้าก่อน rule-based หรือไม่ |
| **Revenue at Risk** คือยอดที่กำลังจะหาย | ตอบ CFO ได้ทันทีว่า "ถ้าไม่ทำอะไร จะเสียเงินเท่าไหร่" |
| **Donut chart** แสดงสัดส่วน 5 กลุ่ม | วางงบ retention ได้ถูกกลุ่ม ไม่หว่านเท่ากันทุกคน |
 
> **การแบ่งกลุ่ม (5 ระดับ):**
> ```
> 🔴 Lost         — หายนานเกิน 3x รอบซื้อปกติ
> 🟥 High Risk    — AI ทำนาย > 75%
> 🟧 Warning      — ช้ากว่ารอบปกติ 1.5x
> 🟨 Medium Risk  — AI ทำนาย 40–75%
> 🟩 Active       — ปกติดี
> ```
 
---
