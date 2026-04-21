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
# 0. LANGUAGE SYSTEM (i18n)
# ==========================================
TRANSLATIONS = {
    "th": {
        # Sidebar
        "app_title": "✈️ Olist Cockpit",
        "refresh_data": "🔄 รีเฟรชข้อมูล",
        "data_loaded": "✅ โหลดข้อมูลแล้ว ({n} rows)",
        "model_threshold": "🎯 Model Threshold: {v}",
        "business_rules": "**📊 Business Rules:**",
        "rule_lost": "🔴 Lost: Late > 3.0",
        "rule_high": "🟥 High Risk: AI > 75%",
        "rule_warning": "🟧 Warning: Late > 1.5",
        "rule_medium": "🟨 Medium Risk: AI 40–75%",
        "rule_active": "🟩 Active: AI < 40%",
        "navigation": "Navigation",
        "language": "Language / ภาษา",
        # Pages
        "page_business": "1. 💰 Business Overview",
        "page_churn": "2. 📊 Churn Overview",
        "page_action": "3. 🎯 Action Plan",
        "page_cycle": "4. 🔄 Buying Cycle Analysis",
        "page_logistics": "5. 🚛 Logistics Insights",
        "page_seller": "6. 🏪 Seller Audit",
        "page_customer": "7. 🔍 Customer Detail",
        # Page 1
        "p1_title": "💰 Business Overview",
        "p1_caption": "ภาพรวมรายได้และสุขภาพธุรกิจ",
        "p1_filter": "🌪️ กรองข้อมูล",
        "p1_cat_label": "หมวดสินค้า (ว่าง = ทั้งหมด):",
        "p1_metric_revenue": "💰 Total Revenue",
        "p1_metric_mom": "📈 MoM Growth",
        "p1_metric_aov": "🛒 Avg Order Value",
        "p1_trend_title": "📈 Monthly Revenue Trend",
        "p1_no_data": "ไม่มีข้อมูลเพียงพอ",
        "p1_top_cat": "🏆 หมวดสินค้าขายดี (Top Categories)",
        "p1_table_header": "**📋 รายละเอียดทุกหมวด**",
        "p1_chart_title": "Top 20 หมวดสินค้า (สีแดง = Churn Risk สูง)",
        "p1_col_cat": "หมวดสินค้า",
        "p1_col_revenue": "Revenue (R$)",
        "p1_col_orders": "Orders",
        "p1_col_avg": "Avg Order (R$)",
        "p1_col_churn": "Churn Risk",
        # Page 2
        "p2_title": "📊 Churn Overview",
        "p2_segment_info": "ℹ️ วิธีแบ่งกลุ่มลูกค้า",
        "p2_filter": "🌪️ กรองข้อมูล",
        "p2_cat_label": "หมวดสินค้า:",
        "p2_metric_atrisk": "🚨 At-Risk",
        "p2_metric_ai": "🤖 AI Predicted",
        "p2_metric_rev": "💸 Revenue at Risk",
        "p2_metric_ratio": "👥 Risk / Total",
        "p2_metric_cycle": "🔄 Avg Cycle",
        "p2_trend": "📈 Churn Risk Trend",
        "p2_rev_risk": "💰 Revenue by Risk",
        "p2_no_data": "ข้อมูลไม่เพียงพอสำหรับ Trend",
        "p2_cycle_unit": " วัน",
        "p2_na": "N/A",
        "p2_rule_label": "Rule-based (%)",
        "p2_ai_label": "AI Predicted (%)",
        "p2_timeline": "Timeline",
        "p2_churn_rate": "Churn Rate (%)",
        # Page 3
        "p3_title": "🎯 Action Plan & Simulator",
        "p3_caption": "จำลองผลกระทบโดยเปลี่ยนฟีเจอร์ → ทำนายซ้ำด้วยโมเดล → วัด Uplift จริง",
        "p3_target": "🎯 กำหนดกลุ่มเป้าหมาย",
        "p3_risk_seg": "กลุ่มความเสี่ยง:",
        "p3_cat_label": "หมวดสินค้า (ว่าง = ทุกหมวด):",
        "p3_analyzing": "📊 กำลังวิเคราะห์: **{g}**",
        "p3_target_pop": "👥 กลุ่มเป้าหมาย",
        "p3_avg_ltv": "💰 LTV เฉลี่ย/คน",
        "p3_people": " คน",
        "p3_problem": "**📉 ปัญหา:** พบ {n} คน\n({pct:.1f}% ของกลุ่มนี้)",
        "p3_feature_avg": "**📋 Feature เฉลี่ย:**",
        "p3_strategy": "**🛠️ วิธีแก้ไข: {name}**",
        "p3_cost_label": "งบต่อหัว (R$)",
        "p3_breakeven": "📐 จุดคุ้มทุน: ต้องสำเร็จ ≥ **{r:.1%}**",
        "p3_ai_pred": "**🤖 AI Prediction:** `{r}%`",
        "p3_no_model": "(โมเดลไม่พร้อม → ใช้ค่าประมาณ)",
        "p3_lift_label": "ปรับค่าคาดการณ์ความสำเร็จ (%)",
        "p3_simulating": "⚡ โมเดลกำลังจำลอง...",
        "p3_results": "**🚀 ผลลัพธ์**",
        "p3_success_rate": "🤖 Success Rate (โมเดล)",
        "p3_breakeven_delta": "จุดคุ้มทุน {r:.1%}",
        "p3_saved": "👥 ดึงลูกค้าคืน",
        "p3_budget": "💸 งบประมาณ",
        "p3_profit": "📈 กำไรสุทธิ (ROI)",
        "p3_loss": "📉 ขาดทุนสุทธิ",
        "p3_worthit": "✅ **คุ้มค่าการลงทุน!**",
        "p3_notworth": "⚠️ **ขาดทุน!**\n\nต้องการ Success Rate: **{be:.1%}**\nได้จริง: **{sr:.1%}**\nขาดอีก: **{gap:.1%}**",
        "p3_reduce_cost": "💡 ลดงบต่อหัวเหลือ **R$ {c:.0f}** เพื่อเริ่มกำไร",
        "p3_uplift_chart": "📊 Uplift Distribution",
        "p3_high_resp": "ตอบสนองสูง\n(>15%)",
        "p3_mid_resp": "ปานกลาง\n(8–15%)",
        "p3_low_resp": "ต่ำ\n(0–8%)",
        "p3_no_resp": "ไม่ตอบสนอง",
        "p3_tab1": "🚚 1. ส่งฟรี / ลดค่าส่ง",
        "p3_tab2": "💵 2. ส่วนลดสินค้า",
        "p3_tab3": "❤️ 3. ง้อลูกค้าส่งช้า",
        "p3_tab4": "🛍️ 4. ขายพ่วง / Cross-sell",
        "p3_tab1_title": "🚚 กลุ่มค่าส่งแพงเกินรับไหว (Freight Pain)",
        "p3_tab1_strategy": "ส่งฟรี (Free Shipping)",
        "p3_tab1_rec": "ลูกค้าลังเลเพราะค่าส่งแพง (เฉลี่ย R$ {avg:.0f})\n\n👉 **Action:** ตั้ง `freight_value = 0` แล้วให้โมเดลทำนายซ้ำ",
        "p3_tab2_title": "💵 กลุ่มเสี่ยง Churn (Price Sensitivity)",
        "p3_tab2_disc": "เลือก % ส่วนลด:",
        "p3_tab2_strategy": "ส่วนลดสินค้า {d}%",
        "p3_tab2_rec": "ลด `price` ลง {d}% แล้วให้โมเดลทำนายซ้ำ\n\n👉 **Action:** เสนอ Coupon {d}% เฉพาะลูกค้า churn_prob > 50%",
        "p3_tab3_title": "❤️ กลุ่มโดนเท / ของส่งช้า (Delay Recovery)",
        "p3_tab3_strategy": "SMS ขอโทษ + คูปองชดเชย",
        "p3_tab3_rec": "ตั้ง `delay_days = 0` (สมมติว่าปัญหาได้รับการแก้ไข)\n\n👉 **Action:** ส่ง SMS ขอโทษทันที + แนบ Coupon ส่วนลดพิเศษ",
        "p3_tab4_title": "🛍️ กลุ่มซื้อหมวดเสี่ยง Churn สูง",
        "p3_tab4_strategy": "Cross-sell + ผ่อนได้นานขึ้น",
        "p3_tab4_rec": "ลด `cat_churn_risk` ลง 40% (จาก cross-sell หมวดซื้อซ้ำ)\n\n👉 **Action:** ยิงแอดสินค้า Housewares + เพิ่ม installments",
        "p3_no_freight": "ไม่พบข้อมูล freight_ratio",
        "p3_no_price": "ไม่พบข้อมูล price",
        "p3_no_delay": "ไม่พบข้อมูล delay_days",
        "p3_no_cat": "ไม่พบข้อมูล cat_churn_risk",
        "p3_all_groups": "ทุกกลุ่ม",
        # Page 4 (Cycle)
        "p4_title": "🔄 Buying Cycle Analysis",
        "p4_cat_label": "📦 หมวดสินค้า:",
        "p4_avg_cycle": "⏱️ รอบซื้อเฉลี่ย",
        "p4_avg_cycle_delta": " วัน vs ภาพรวม",
        "p4_lateness": "🐢 ความล่าช้า",
        "p4_fast": "📅 ซื้อซ้ำใน 30 วัน",
        "p4_trend_title": "📈 Buying Cycle Trend",
        "p4_no_trend": "ข้อมูลไม่เพียงพอสำหรับ Trend",
        "p4_no_repeat": "ไม่พบลูกค้าที่ซื้อซ้ำ",
        "p4_detail": "📋 รายละเอียดรายหมวด",
        "p4_col_cust": "ลูกค้า",
        "p4_col_cycle": "รอบซื้อ",
        "p4_col_late": "ความล่าช้า",
        "p4_heatmap": "📅 Seasonal Heatmap",
        "p4_heat_tip": "💡 สีส้มเข้ม = High Season → เตรียมสต็อกล่วงหน้า",
        "p4_heat_month": "เดือน",
        "p4_heat_cat": "หมวด",
        "p4_heat_vol": "ยอดขาย",
        "p4_gap_y": "ระยะเวลาซื้อซ้ำเฉลี่ย (วัน)",
        "p4_gap_tooltip": "วัน",
        "p4_people_unit": " คน",
        "p4_days_unit": " วัน",
        # Page 5 (Logistics)
        "p5_title": "🚛 Logistics Insights",
        "p5_no_state": "❌ ไม่พบ customer_state",
        "p5_cat_label": "📦 หมวดสินค้า:",
        "p5_status_label": "👥 สถานะ:",
        "p5_focus": "🔍 โฟกัสรัฐ:",
        "p5_metric_rev": "💰 ยอดเงิน",
        "p5_metric_del": "🚚 ส่งเฉลี่ย",
        "p5_metric_late": "⚠️ ส่งช้า",
        "p5_map_title": "🗺️ แผนที่ ({z})",
        "p5_issues": "🚨 Top Issues",
        "p5_sort": "เรียงตาม:",
        "p5_sort_late": "ส่งช้า",
        "p5_sort_risk": "ความเสี่ยง",
        "p5_col_money": "เงิน",
        "p5_col_days": "ส่ง(วัน)",
        "p5_city_title": "🏙️ เจาะลึกรายเมือง",
        "p5_city_info_all": "📍 ทั่วประเทศ — Top 50 ที่ส่งช้ามากสุด",
        "p5_city_info_state": "📍 รัฐ {s}",
        "p5_col_cust": "ลูกค้า",
        "p5_col_revenue": "ยอดเงิน",
        "p5_col_late2": "ส่งช้า",
        "p5_times": " ครั้ง",
        "p5_days": " วัน",
        # Page 6 (Seller)
        "p6_title": "🏪 Seller Audit",
        "p6_no_seller": "❌ ไม่พบ seller_id",
        "p6_cat_label": "📦 หมวดสินค้า:",
        "p6_status_label": "👥 สถานะ:",
        "p6_metric_shops": "🏪 ร้านค้า",
        "p6_metric_rev": "💸 ยอดขายรวม",
        "p6_metric_review": "⭐ รีวิวเฉลี่ย",
        "p6_metric_del": "🚚 ส่งเฉลี่ย",
        "p6_sort": "เรียงตาม:",
        "p6_sort_risk": "🚨 ความเสี่ยง",
        "p6_sort_late": "🐢 ส่งช้า",
        "p6_sort_score": "⭐ คะแนนต่ำ",
        "p6_sort_rev": "💸 ยอดขาย",
        "p6_sort_vol": "📦 ปริมาณ",
        "p6_col_del": "ส่ง(วัน)",
        "p6_na": "N/A",
        "p6_days": " วัน",
        # Page 7 (Customer)
        "p7_title": "🔍 Customer Deep Dive",
        "p7_filters": "🔎 Filters",
        "p7_status_label": "สถานะ:",
        "p7_cat_label": "หมวดสินค้า:",
        "p7_search": "ค้นหา Customer ID:",
        "p7_top10": "📊 Top 10 หมวดเสี่ยง",
        "p7_detail": "📋 รายละเอียด",
        "p7_list": "📄 รายชื่อลูกค้า ({n} คน)",
        "p7_col_late": "Late",
        # Statuses
        "status_high": "High Risk",
        "status_warning": "Warning (Late > 1.5x)",
        "status_medium": "Medium Risk",
        "status_lost": "Lost (Late > 3x)",
        "status_active": "Active",
        # Segment table
        "seg_condition": "เงื่อนไข",
        "seg_status": "สถานะ",
        "seg_lost_cond": "Lateness > 3.0",
        "seg_high_cond": "AI > 75%",
        "seg_warn_cond": "Lateness > 1.5",
        "seg_med_cond": "AI 40–75% (ขยายแล้ว!)",
        "seg_act_cond": "AI < 40%",
    },
    "en": {
        # Sidebar
        "app_title": "✈️ Olist Cockpit",
        "refresh_data": "🔄 Refresh Data",
        "data_loaded": "✅ Data loaded ({n} rows)",
        "model_threshold": "🎯 Model Threshold: {v}",
        "business_rules": "**📊 Business Rules:**",
        "rule_lost": "🔴 Lost: Late > 3.0",
        "rule_high": "🟥 High Risk: AI > 75%",
        "rule_warning": "🟧 Warning: Late > 1.5",
        "rule_medium": "🟨 Medium Risk: AI 40–75%",
        "rule_active": "🟩 Active: AI < 40%",
        "navigation": "Navigation",
        "language": "Language / ภาษา",
        # Pages
        "page_business": "1. 💰 Business Overview",
        "page_churn": "2. 📊 Churn Overview",
        "page_action": "3. 🎯 Action Plan",
        "page_cycle": "4. 🔄 Buying Cycle Analysis",
        "page_logistics": "5. 🚛 Logistics Insights",
        "page_seller": "6. 🏪 Seller Audit",
        "page_customer": "7. 🔍 Customer Detail",
        # Page 1
        "p1_title": "💰 Business Overview",
        "p1_caption": "Revenue and business health summary",
        "p1_filter": "🌪️ Filter Data",
        "p1_cat_label": "Product Category (empty = all):",
        "p1_metric_revenue": "💰 Total Revenue",
        "p1_metric_mom": "📈 MoM Growth",
        "p1_metric_aov": "🛒 Avg Order Value",
        "p1_trend_title": "📈 Monthly Revenue Trend",
        "p1_no_data": "Insufficient data",
        "p1_top_cat": "🏆 Top Selling Categories",
        "p1_table_header": "**📋 All Category Details**",
        "p1_chart_title": "Top 20 Categories (Red = High Churn Risk)",
        "p1_col_cat": "Category",
        "p1_col_revenue": "Revenue (R$)",
        "p1_col_orders": "Orders",
        "p1_col_avg": "Avg Order (R$)",
        "p1_col_churn": "Churn Risk",
        # Page 2
        "p2_title": "📊 Churn Overview",
        "p2_segment_info": "ℹ️ Customer Segmentation Guide",
        "p2_filter": "🌪️ Filter Data",
        "p2_cat_label": "Product Category:",
        "p2_metric_atrisk": "🚨 At-Risk",
        "p2_metric_ai": "🤖 AI Predicted",
        "p2_metric_rev": "💸 Revenue at Risk",
        "p2_metric_ratio": "👥 Risk / Total",
        "p2_metric_cycle": "🔄 Avg Cycle",
        "p2_trend": "📈 Churn Risk Trend",
        "p2_rev_risk": "💰 Revenue by Risk",
        "p2_no_data": "Insufficient data for trend",
        "p2_cycle_unit": " days",
        "p2_na": "N/A",
        "p2_rule_label": "Rule-based (%)",
        "p2_ai_label": "AI Predicted (%)",
        "p2_timeline": "Timeline",
        "p2_churn_rate": "Churn Rate (%)",
        # Page 3
        "p3_title": "🎯 Action Plan & Simulator",
        "p3_caption": "Simulate impact by modifying features → Re-predict with model → Measure real Uplift",
        "p3_target": "🎯 Define Target Audience",
        "p3_risk_seg": "Risk Segments:",
        "p3_cat_label": "Product Category (empty = all):",
        "p3_analyzing": "📊 Analyzing: **{g}**",
        "p3_target_pop": "👥 Target Population",
        "p3_avg_ltv": "💰 Avg LTV/person",
        "p3_people": " people",
        "p3_problem": "**📉 Problem:** {n} customers found\n({pct:.1f}% of this group)",
        "p3_feature_avg": "**📋 Avg Feature Values:**",
        "p3_strategy": "**🛠️ Solution: {name}**",
        "p3_cost_label": "Cost per person (R$)",
        "p3_breakeven": "📐 Break-even: need ≥ **{r:.1%}** success",
        "p3_ai_pred": "**🤖 AI Prediction:** `{r}%`",
        "p3_no_model": "(Model unavailable → using estimate)",
        "p3_lift_label": "Adjust estimated success rate (%)",
        "p3_simulating": "⚡ Model simulating...",
        "p3_results": "**🚀 Results**",
        "p3_success_rate": "🤖 Success Rate (Model)",
        "p3_breakeven_delta": "Break-even {r:.1%}",
        "p3_saved": "👥 Customers Retained",
        "p3_budget": "💸 Budget",
        "p3_profit": "📈 Net Profit (ROI)",
        "p3_loss": "📉 Net Loss",
        "p3_worthit": "✅ **Profitable investment!**",
        "p3_notworth": "⚠️ **Loss-making!**\n\nRequired Success Rate: **{be:.1%}**\nActual: **{sr:.1%}**\nGap: **{gap:.1%}**",
        "p3_reduce_cost": "💡 Reduce cost/person to **R$ {c:.0f}** to break even",
        "p3_uplift_chart": "📊 Uplift Distribution",
        "p3_high_resp": "High Response\n(>15%)",
        "p3_mid_resp": "Medium\n(8–15%)",
        "p3_low_resp": "Low\n(0–8%)",
        "p3_no_resp": "No Response",
        "p3_tab1": "🚚 1. Free / Discounted Shipping",
        "p3_tab2": "💵 2. Product Discount",
        "p3_tab3": "❤️ 3. Win-Back Late Delivery",
        "p3_tab4": "🛍️ 4. Upsell / Cross-sell",
        "p3_tab1_title": "🚚 High Freight Cost Group (Freight Pain)",
        "p3_tab1_strategy": "Free Shipping",
        "p3_tab1_rec": "Customers hesitating due to high freight (avg R$ {avg:.0f})\n\n👉 **Action:** Set `freight_value = 0` and re-predict with model",
        "p3_tab2_title": "💵 Churn Risk Group (Price Sensitivity)",
        "p3_tab2_disc": "Select discount %:",
        "p3_tab2_strategy": "{d}% Product Discount",
        "p3_tab2_rec": "Reduce `price` by {d}% and re-predict\n\n👉 **Action:** Offer {d}% Coupon to customers with churn_prob > 50%",
        "p3_tab3_title": "❤️ Late Delivery Recovery Group",
        "p3_tab3_strategy": "Apology SMS + Compensation Coupon",
        "p3_tab3_rec": "Set `delay_days = 0` (simulate problem resolved)\n\n👉 **Action:** Send immediate apology SMS + attach special discount coupon",
        "p3_tab4_title": "🛍️ High Churn Risk Category Buyers",
        "p3_tab4_strategy": "Cross-sell + Extended Installments",
        "p3_tab4_rec": "Reduce `cat_churn_risk` by 40% (via cross-sell to repeat categories)\n\n👉 **Action:** Target Housewares ads + add more installment options",
        "p3_no_freight": "freight_ratio data not found",
        "p3_no_price": "price data not found",
        "p3_no_delay": "delay_days data not found",
        "p3_no_cat": "cat_churn_risk data not found",
        "p3_all_groups": "All groups",
        # Page 4 (Cycle)
        "p4_title": "🔄 Buying Cycle Analysis",
        "p4_cat_label": "📦 Product Category:",
        "p4_avg_cycle": "⏱️ Avg Buying Cycle",
        "p4_avg_cycle_delta": " days vs overall",
        "p4_lateness": "🐢 Lateness Score",
        "p4_fast": "📅 Repurchase ≤ 30 days",
        "p4_trend_title": "📈 Buying Cycle Trend",
        "p4_no_trend": "Insufficient data for trend",
        "p4_no_repeat": "No repeat buyers found",
        "p4_detail": "📋 Category Details",
        "p4_col_cust": "Customers",
        "p4_col_cycle": "Cycle",
        "p4_col_late": "Lateness",
        "p4_heatmap": "📅 Seasonal Heatmap",
        "p4_heat_tip": "💡 Dark orange = High Season → Prepare stock in advance",
        "p4_heat_month": "Month",
        "p4_heat_cat": "Category",
        "p4_heat_vol": "Volume",
        "p4_gap_y": "Avg Repurchase Gap (days)",
        "p4_gap_tooltip": "days",
        "p4_people_unit": " people",
        "p4_days_unit": " days",
        # Page 5 (Logistics)
        "p5_title": "🚛 Logistics Insights",
        "p5_no_state": "❌ customer_state column not found",
        "p5_cat_label": "📦 Product Category:",
        "p5_status_label": "👥 Status:",
        "p5_focus": "🔍 Focus State:",
        "p5_metric_rev": "💰 Revenue",
        "p5_metric_del": "🚚 Avg Delivery",
        "p5_metric_late": "⚠️ Late Deliveries",
        "p5_map_title": "🗺️ Map ({z})",
        "p5_issues": "🚨 Top Issues",
        "p5_sort": "Sort by:",
        "p5_sort_late": "Late Delivery",
        "p5_sort_risk": "Risk",
        "p5_col_money": "Revenue",
        "p5_col_days": "Del.(days)",
        "p5_city_title": "🏙️ City-Level Insights",
        "p5_city_info_all": "📍 Nationwide — Top 50 most late deliveries",
        "p5_city_info_state": "📍 State: {s}",
        "p5_col_cust": "Customers",
        "p5_col_revenue": "Revenue",
        "p5_col_late2": "Late",
        "p5_times": " times",
        "p5_days": " days",
        # Page 6 (Seller)
        "p6_title": "🏪 Seller Audit",
        "p6_no_seller": "❌ seller_id column not found",
        "p6_cat_label": "📦 Product Category:",
        "p6_status_label": "👥 Status:",
        "p6_metric_shops": "🏪 Sellers",
        "p6_metric_rev": "💸 Total Revenue",
        "p6_metric_review": "⭐ Avg Review",
        "p6_metric_del": "🚚 Avg Delivery",
        "p6_sort": "Sort by:",
        "p6_sort_risk": "🚨 Risk",
        "p6_sort_late": "🐢 Slow Delivery",
        "p6_sort_score": "⭐ Low Score",
        "p6_sort_rev": "💸 Revenue",
        "p6_sort_vol": "📦 Volume",
        "p6_col_del": "Del.(days)",
        "p6_na": "N/A",
        "p6_days": " days",
        # Page 7 (Customer)
        "p7_title": "🔍 Customer Deep Dive",
        "p7_filters": "🔎 Filters",
        "p7_status_label": "Status:",
        "p7_cat_label": "Product Category:",
        "p7_search": "Search Customer ID:",
        "p7_top10": "📊 Top 10 Risk Categories",
        "p7_detail": "📋 Details",
        "p7_list": "📄 Customer List ({n} people)",
        "p7_col_late": "Late",
        # Statuses
        "status_high": "High Risk",
        "status_warning": "Warning (Late > 1.5x)",
        "status_medium": "Medium Risk",
        "status_lost": "Lost (Late > 3x)",
        "status_active": "Active",
        # Segment table
        "seg_condition": "Condition",
        "seg_status": "Status",
        "seg_lost_cond": "Lateness > 3.0",
        "seg_high_cond": "AI > 75%",
        "seg_warn_cond": "Lateness > 1.5",
        "seg_med_cond": "AI 40–75% (expanded!)",
        "seg_act_cond": "AI < 40%",
    }
}

# ── Language state ────────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "th"

def t(key, **kwargs):
    """Translate key with optional format args."""
    text = TRANSLATIONS[st.session_state.lang].get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text

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
    div[data-testid="stRadio"] > label { font-size: 0.9rem; }
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

    if 'freight_value' in df.columns and 'price' in df.columns:
        df['freight_ratio'] = np.where(
            df['price'] > 0, df['freight_value'] / df['price'], 0)
        df['payment_value'] = df['price'] + df['freight_value']
    else:
        df['freight_ratio'] = 0
        df['payment_value'] = df.get('price', 0)

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

    if 'review_score' in df.columns:
        df['review_score']  = pd.to_numeric(df['review_score'], errors='coerce')
        df['is_low_score']  = (df['review_score'].fillna(3) <= 2).astype(int)
        df['is_high_score'] = (df['review_score'].fillna(3) == 5).astype(int)
    else:
        df['review_score']  = 3.0
        df['is_low_score']  = 0
        df['is_high_score'] = 0

    df['purchase_count']    = df.groupby('customer_unique_id').cumcount() + 1
    df['is_first_purchase'] = (df['purchase_count'] == 1).astype(int)
    df['is_repeat_buyer']   = (df['purchase_count'] >= 2).astype(int)

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

    if 'cat_churn_risk' not in df.columns:
        df['cat_churn_risk'] = 0.80

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
    # ── Language toggle ───────────────────────────────────────────────
    st.markdown(f"**{t('language')}**")
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        if st.button("🇹🇭 ไทย", use_container_width=True,
                     type="primary" if st.session_state.lang == "th" else "secondary"):
            st.session_state.lang = "th"
            st.rerun()
    with lang_col2:
        if st.button("🇬🇧 English", use_container_width=True,
                     type="primary" if st.session_state.lang == "en" else "secondary"):
            st.session_state.lang = "en"
            st.rerun()
    st.markdown("---")

    if st.button(t("refresh_data")):
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
# 9. NAVIGATION (continued sidebar)
# ==========================================
with st.sidebar:
    st.title(t("app_title"))
    st.success(t("data_loaded", n=f"{len(df):,}"))
    st.info(t("model_threshold", v=f"{best_threshold:.2f}"))
    st.markdown(t("business_rules"))
    st.markdown(f"- {t('rule_lost')}")
    st.markdown(f"- {t('rule_high')}")
    st.markdown(f"- {t('rule_warning')}")
    st.markdown(f"- {t('rule_medium')}")
    st.markdown(f"- {t('rule_active')}")

    page = st.radio(t("navigation"), [
        t("page_business"),
        t("page_churn"),
        t("page_action"),
        t("page_cycle"),
        t("page_logistics"),
        t("page_seller"),
        t("page_customer"),
    ])
    st.markdown("---")

# helper
def safe_cats(dataframe, col='product_category_name'):
    if col not in dataframe.columns: return []
    return sorted([x for x in dataframe[col].unique() if pd.notna(x)])

# ==========================================
# PAGE 1: Business Overview
# ==========================================
if page == t("page_business"):
    st.title(t("p1_title"))
    st.caption(t("p1_caption"))

    with st.expander(t("p1_filter"), expanded=False):
        sel_cats = st.multiselect(t("p1_cat_label"), safe_cats(df), key="p1_cat")

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
    k1.metric(t("p1_metric_revenue"), f"R$ {total_rev:,.0f}")
    k2.metric(t("p1_metric_mom"), f"{mom_growth:+.1f}%" if mom_growth is not None else "N/A",
              delta=f"{mom_growth:+.1f}%" if mom_growth is not None else None)
    k3.metric(t("p1_metric_aov"), f"R$ {avg_order:,.0f}")
    st.markdown("---")

    st.subheader(t("p1_trend_title"))
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
        st.info(t("p1_no_data"))

    st.markdown("---")
    st.subheader(t("p1_top_cat"))
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
                    title=t("p1_col_churn")),
                tooltip=[
                    alt.Tooltip('product_category_name', title=t("p1_col_cat")),
                    alt.Tooltip('revenue', format=',.0f', title='Revenue (R$)'),
                    alt.Tooltip('orders', format=','),
                    alt.Tooltip('churn_risk', format='.1%', title=t("p1_col_churn")),
                ]
            ).properties(height=500, title=t("p1_chart_title"))
            st.altair_chart(bar_cat, use_container_width=True)

        with col_table:
            st.markdown(t("p1_table_header"))
            st.dataframe(
                cat_sales.rename(columns={
                    'product_category_name': t("p1_col_cat"),
                    'revenue':   t("p1_col_revenue"),
                    'orders':    t("p1_col_orders"),
                    'avg_order': t("p1_col_avg"),
                    'churn_risk': t("p1_col_churn")
                }),
                column_config={
                    t("p1_col_revenue"):  st.column_config.NumberColumn(format='R$ %.0f'),
                    t("p1_col_orders"):   st.column_config.NumberColumn(format='%,d'),
                    t("p1_col_avg"):      st.column_config.NumberColumn(format='R$ %.0f'),
                    t("p1_col_churn"):    st.column_config.ProgressColumn(
                        format='%.2f', min_value=0, max_value=1),
                },
                use_container_width=True,
                hide_index=True,
                height=500
            )

# ==========================================
# PAGE 2: Churn Overview
# ==========================================
elif page == t("page_churn"):
    st.title(t("p2_title"))

    with st.expander(t("p2_segment_info"), expanded=True):
        st.markdown(f"""
| {t('seg_status')} | {t('seg_condition')} |
|---|---|
| 🔴 Lost | {t('seg_lost_cond')} |
| 🟥 High Risk | {t('seg_high_cond')} |
| 🟧 Warning | {t('seg_warn_cond')} |
| 🟨 **Medium Risk** | **{t('seg_med_cond')}** |
| 🟩 Active | {t('seg_act_cond')} |
        """)

    with st.expander(t("p2_filter"), expanded=False):
        sel_cats = st.multiselect(t("p2_cat_label"), safe_cats(df), key="p2_cat")

    df_d = df[df['product_category_name'].isin(sel_cats)].copy() if sel_cats else df.copy()
    st.markdown("---")

    total   = len(df_d)
    risk_df = df_d[df_d['status'].isin(['High Risk', 'Warning (Late > 1.5x)'])]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(t("p2_metric_atrisk"),
              f"{len(risk_df)/total*100:.1f}%" if total else "0%")
    k2.metric(t("p2_metric_ai"),
              f"{(df_d['churn_probability'] >= best_threshold).mean()*100:.1f}%")
    k3.metric(t("p2_metric_rev"),
              f"R$ {risk_df['payment_value'].sum():,.0f}")
    k4.metric(t("p2_metric_ratio"),
              f"{len(risk_df):,} / {total:,}")
    k5.metric(t("p2_metric_cycle"),
              f"{df_d['cat_median_days'].mean():.0f}{t('p2_cycle_unit')}"
              if 'cat_median_days' in df_d.columns else t("p2_na"))
    st.markdown("---")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader(t("p2_trend"))
        if 'order_purchase_timestamp' in df_d.columns and not df_d.empty:
            df_d['month_year'] = df_d['order_purchase_timestamp'].dt.to_period('M')
            trend_data = []
            for name, grp in df_d.groupby('month_year'):
                tot = len(grp)
                if tot == 0: continue
                rule = len(grp[grp['status'].isin(['High Risk', 'Warning (Late > 1.5x)'])])
                ai   = (grp['churn_probability'] >= best_threshold).sum()
                trend_data.append({'Date': str(name),
                                   t('p2_rule_label'): rule/tot*100,
                                   t('p2_ai_label'): ai/tot*100})
            tdf = pd.DataFrame(trend_data)
            if len(tdf) > 1:
                tdf = tdf.iloc[:-1]
                tdf['Date'] = pd.to_datetime(tdf['Date'])
                melted = tdf.melt('Date', var_name='Type', value_name='Rate (%)')
                chart = alt.Chart(melted).mark_line(point=True).encode(
                    x=alt.X('Date', axis=alt.Axis(format='%b %Y',
                                                   title=t('p2_timeline'))),
                    y=alt.Y('Rate (%)', title=t('p2_churn_rate')),
                    color=alt.Color('Type', scale=alt.Scale(
                        domain=[t('p2_rule_label'), t('p2_ai_label')],
                        range=['#e67e22', '#8e44ad']),
                        legend=alt.Legend(orient='bottom')),
                    tooltip=['Date', 'Type', alt.Tooltip('Rate (%)', format='.1f')]
                ).properties(height=350)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info(t("p2_no_data"))

    with c2:
        st.subheader(t("p2_rev_risk"))
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
elif page == t("page_action"):
    st.title(t("p3_title"))
    st.caption(t("p3_caption"))

    status_options = [
        t("status_high"), t("status_warning"),
        t("status_medium"), t("status_lost")
    ]
    # Internal status values (always English)
    status_internal = ['High Risk', 'Warning (Late > 1.5x)', 'Medium Risk', 'Lost (Late > 3x)']
    status_map = dict(zip(status_options, status_internal))

    with st.expander(t("p3_target"), expanded=True):
        f1, f2 = st.columns(2)
        with f1:
            risk_segments_display = st.multiselect(
                t("p3_risk_seg"), status_options,
                default=status_options[:2]
            )
            risk_segments = [status_map[s] for s in risk_segments_display]
        with f2:
            sel_cats_p3 = st.multiselect(t("p3_cat_label"),
                                         safe_cats(df), key="p3_cat_multiselect")

    df_p3 = df.copy()
    if risk_segments:
        df_p3 = df_p3[df_p3['status'].isin(risk_segments)]
    if sel_cats_p3:
        df_p3 = df_p3[df_p3['product_category_name'].isin(sel_cats_p3)]

    filter_msg = (f"{', '.join(risk_segments_display[:2])}"
                  f"{'...' if len(risk_segments_display) > 2 else ''}") \
                  if risk_segments_display else t("p3_all_groups")
    total_pop = len(df_p3)
    avg_ltv   = float(df_p3['payment_value'].mean()) if 'payment_value' in df_p3.columns else 150.0

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: st.info(t("p3_analyzing", g=filter_msg))
    with c2: st.metric(t("p3_target_pop"), f"{total_pop:,}{t('p3_people')}")
    with c3: st.metric(t("p3_avg_ltv"), f"R$ {avg_ltv:,.0f}")
    st.markdown("---")

    def run_simulation(target_df, feature_changes: dict, cost_per_head: float,
                       tab_key: str, rec_text: str, strategy_name: str):
        n_target    = len(target_df)
        pct_problem = (n_target / total_pop * 100) if total_pop > 0 else 0

        c_prob, c_sol, c_res = st.columns([1, 1.3, 1])

        with c_prob:
            st.info(t("p3_problem", n=f"{n_target:,}", pct=pct_problem))
            st.progress(min(pct_problem / 100, 1.0))
            if not target_df.empty:
                st.markdown(t("p3_feature_avg"))
                for col in list(feature_changes.keys())[:3]:
                    if col in target_df.columns:
                        st.caption(f"• {col}: {target_df[col].mean():.2f}")

        with c_sol:
            st.markdown(t("p3_strategy", name=strategy_name))
            st.write(rec_text)
            st.markdown("---")
            cost = st.number_input(
                t("p3_cost_label"), value=float(cost_per_head),
                min_value=0.0, max_value=500.0, step=0.5,
                key=f"cost_{tab_key}"
            )
            break_even_rate = cost / avg_ltv if avg_ltv > 0 else 0
            st.caption(t("p3_breakeven", r=break_even_rate))

            if model is None or not feature_names:
                max_pot   = 15
                realistic = min(max_pot, 10) if cost >= 15 else min(max_pot, 5)
                st.markdown(t("p3_ai_pred", r=realistic))
                st.caption(t("p3_no_model"))
                lift = st.slider(t("p3_lift_label"), 1, 100,
                                 realistic, key=f"lift_{tab_key}")
                sim_success_rate = lift / 100
                sim_mode = "manual"
            else:
                sim_mode = "model"
                lift     = None

        with c_res:
            with st.spinner(t("p3_simulating")):
                time.sleep(0.3)

                if sim_mode == "model" and not target_df.empty:
                    X_orig    = target_df.reindex(columns=feature_names, fill_value=0).fillna(0)
                    prob_orig = model.predict_proba(X_orig)[:, 1]

                    df_sim = target_df.copy()
                    for col, (op, val) in feature_changes.items():
                        if col in df_sim.columns:
                            if op == 'set':          df_sim[col] = val
                            elif op == 'multiply':   df_sim[col] = df_sim[col] * val
                            elif op == 'clip_upper': df_sim[col] = df_sim[col].clip(upper=val)
                            elif op == 'add':        df_sim[col] = df_sim[col] + val

                    if 'freight_value' in df_sim.columns and 'price' in df_sim.columns:
                        df_sim['freight_ratio'] = (
                            df_sim['freight_value'] /
                            df_sim['price'].replace(0, np.nan)
                        ).fillna(0)

                    X_sim    = df_sim.reindex(columns=feature_names, fill_value=0).fillna(0)
                    prob_sim = model.predict_proba(X_sim)[:, 1]

                    uplift_arr       = prob_orig - prob_sim
                    THRESHOLD        = 0.08
                    sim_success_rate = (uplift_arr > THRESHOLD).mean()

                    dist = {
                        t("p3_high_resp"): int((uplift_arr > 0.15).sum()),
                        t("p3_mid_resp"):  int(((uplift_arr > 0.08) & (uplift_arr <= 0.15)).sum()),
                        t("p3_low_resp"):  int(((uplift_arr > 0) & (uplift_arr <= 0.08)).sum()),
                        t("p3_no_resp"):   int((uplift_arr <= 0).sum()),
                    }
                    dist_df = pd.DataFrame({"Group": list(dist.keys()),
                                            "Count": list(dist.values())})
                    st.altair_chart(
                        alt.Chart(dist_df).mark_bar().encode(
                            x=alt.X("Group", sort=None, axis=alt.Axis(labelAngle=0)),
                            y=alt.Y("Count"),
                            color=alt.Color("Group", scale=alt.Scale(
                                domain=list(dist.keys()),
                                range=["#2ecc71", "#f1c40f", "#e67e22", "#95a5a6"]
                            ), legend=None),
                            tooltip=["Group", "Count"]
                        ).properties(height=160, title=t("p3_uplift_chart")),
                        use_container_width=True
                    )
                else:
                    sim_success_rate = lift / 100 if lift else 0.1

                budget      = n_target * cost
                saved_users = int(n_target * sim_success_rate)
                revenue     = saved_users * avg_ltv
                profit      = revenue - budget
                roi         = (profit / budget * 100) if budget > 0 else 0
                be_final    = cost / avg_ltv if avg_ltv > 0 else 0

                st.markdown(t("p3_results"))
                st.metric(t("p3_success_rate"), f"{sim_success_rate:.1%}",
                          delta=t("p3_breakeven_delta", r=be_final))
                st.metric(t("p3_saved"),  f"{saved_users:,}{t('p3_people')}")
                st.metric(t("p3_budget"), f"R$ {budget:,.0f}")

                if profit > 0:
                    st.metric(t("p3_profit"), f"R$ {profit:,.0f}", f"+{roi:.1f}%")
                    st.success(t("p3_worthit"))
                else:
                    st.metric(t("p3_loss"), f"R$ {profit:,.0f}", f"{roi:.1f}%")
                    gap = be_final - sim_success_rate
                    st.error(t("p3_notworth", be=be_final, sr=sim_success_rate, gap=gap))
                    max_cost_be = avg_ltv * sim_success_rate
                    st.caption(t("p3_reduce_cost", c=max_cost_be))

    tab1, tab2, tab3, tab4 = st.tabs([
        t("p3_tab1"), t("p3_tab2"), t("p3_tab3"), t("p3_tab4")
    ])

    with tab1:
        st.subheader(t("p3_tab1_title"))
        if 'freight_ratio' in df_p3.columns:
            target_t1   = df_p3[df_p3['freight_ratio'] > 0.2].copy()
            avg_freight = float(target_t1['freight_value'].mean()) \
                          if (not target_t1.empty and 'freight_value' in target_t1.columns) else 15.0
            run_simulation(
                target_df=target_t1,
                feature_changes={'freight_value': ('set', 0), 'freight_ratio': ('set', 0)},
                cost_per_head=avg_freight, tab_key="tab1",
                strategy_name=t("p3_tab1_strategy"),
                rec_text=t("p3_tab1_rec", avg=avg_freight)
            )
        else:
            st.error(t("p3_no_freight"))

    with tab2:
        st.subheader(t("p3_tab2_title"))
        disc_pct = st.radio(t("p3_tab2_disc"), [10, 20], horizontal=True, key="disc_pct_t2")
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
                strategy_name=t("p3_tab2_strategy", d=disc_pct),
                rec_text=t("p3_tab2_rec", d=disc_pct)
            )
        else:
            st.error(t("p3_no_price"))

    with tab3:
        st.subheader(t("p3_tab3_title"))
        if 'delay_days' in df_p3.columns:
            target_t3 = df_p3[df_p3['delay_days'] > 0].copy()
            run_simulation(
                target_df=target_t3,
                feature_changes={
                    'delay_days':            ('set', 0),
                    'delivery_vs_estimated': ('clip_upper', 0),
                },
                cost_per_head=15.0, tab_key="tab3",
                strategy_name=t("p3_tab3_strategy"),
                rec_text=t("p3_tab3_rec")
            )
        else:
            st.error(t("p3_no_delay"))

    with tab4:
        st.subheader(t("p3_tab4_title"))
        if 'cat_churn_risk' in df_p3.columns:
            target_t4 = df_p3[df_p3['cat_churn_risk'] > 0.8].copy()
            run_simulation(
                target_df=target_t4,
                feature_changes={
                    'cat_churn_risk':        ('multiply', 0.6),
                    'payment_installments':  ('add', 2),
                },
                cost_per_head=10.0, tab_key="tab4",
                strategy_name=t("p3_tab4_strategy"),
                rec_text=t("p3_tab4_rec")
            )
        else:
            st.error(t("p3_no_cat"))

# ==========================================
# PAGE 4 (nav order): Logistics Insights
# ==========================================
elif page == t("page_logistics"):
    import pydeck as pdk
    st.title(t("p5_title"))

    if 'customer_state' not in df.columns:
        st.error(t("p5_no_state")); st.stop()

    c1, c2 = st.columns(2)
    with c1:
        sel_c = st.multiselect(t("p5_cat_label"), safe_cats(df), key="p4_cat")
    with c2:
        sel_s = st.multiselect(t("p5_status_label"),
            [t("status_high"), t("status_warning"), t("status_medium"),
             t("status_lost"), t("status_active")], key="p4_status")

    status_reverse = {t(k): k.split("status_")[1].replace("_", " ").title()
                      for k in ["status_high","status_warning","status_medium",
                                "status_lost","status_active"]}
    # simpler direct map
    sel_s_internal = [s for s in
                      ['High Risk','Warning (Late > 1.5x)','Medium Risk',
                       'Lost (Late > 3x)','Active']
                      if t("status_" + {
                          'High Risk':'high','Warning (Late > 1.5x)':'warning',
                          'Medium Risk':'medium','Lost (Late > 3x)':'lost',
                          'Active':'active'}[s]) in sel_s] if sel_s else []

    df_log = df.copy()
    if sel_c:           df_log = df_log[df_log['product_category_name'].isin(sel_c)]
    if sel_s_internal:  df_log = df_log[df_log['status'].isin(sel_s_internal)]

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
        zoom = st.selectbox(t("p5_focus"),
                            ["All"] + sorted(sm['customer_state'].unique()))
    disp     = sm if zoom == "All" else sm[sm['customer_state'] == zoom]
    view_lat = disp['lat'].mean() if zoom != "All" else -14.24
    view_lon = disp['lon'].mean() if zoom != "All" else -51.93
    view_z   = 6 if zoom != "All" else 3.5
    k1.metric(t("p5_metric_rev"),  f"R$ {disp['payment_value'].sum():,.0f}")
    k2.metric(t("p5_metric_del"),  f"{disp['delivery_days'].mean():.1f}{t('p5_days')}")
    k3.metric(t("p5_metric_late"), f"{int(disp['delay_count'].sum()):,}{t('p5_times')}")

    cm_, ct_ = st.columns([2, 1])
    with cm_:
        st.subheader(t("p5_map_title", z=zoom))
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
                           "🚚 {delivery_days}<br/>⚠️ {delay_count}",
                   "style": {"backgroundColor": "steelblue", "color": "white"}}
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon,
                                             zoom=view_z, pitch=20),
            tooltip=tooltip, map_provider='carto', map_style='light'))

    with ct_:
        st.subheader(t("p5_issues"))
        sort_options = [t("p5_sort_late"), t("p5_sort_risk")]
        sort_m = st.radio(t("p5_sort"), sort_options, horizontal=True, key="p4_sort")
        sort_col = 'delay_count' if sort_m == t("p5_sort_late") else 'churn_probability'
        top_i = sm.sort_values(sort_col, ascending=False).head(10)
        st.dataframe(top_i[['customer_state', 'payment_value', 'delivery_days',
                             'delay_count', 'churn_probability']],
            column_config={
                "payment_value": st.column_config.NumberColumn(t("p5_col_money"), format="R$%.0f"),
                "delivery_days": st.column_config.NumberColumn(t("p5_col_days"), format="%.1f"),
                "churn_probability": st.column_config.ProgressColumn(
                    "Risk", format="%.2f", min_value=0, max_value=1)
            }, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader(t("p5_city_title"))
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
        city_info = t("p5_city_info_state", s=zoom) if zoom != "All" else t("p5_city_info_all")
        st.info(city_info)
        st.dataframe(disp_c.sort_values('late', ascending=False).head(50),
            column_config={
                "n": st.column_config.NumberColumn(t("p5_col_cust")),
                "revenue": st.column_config.NumberColumn(t("p5_col_revenue"), format="R$%.0f"),
                "del_days": st.column_config.NumberColumn(t("p5_col_days"), format="%.1f"),
                "late": st.column_config.NumberColumn(t("p5_col_late2")),
                "risk": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1)
            }, hide_index=True, use_container_width=True)

# ==========================================
# PAGE 5 (nav): Seller Audit
# ==========================================
elif page == t("page_seller"):
    st.title(t("p6_title"))

    if 'seller_id' not in df.columns:
        st.error(t("p6_no_seller")); st.stop()

    c1, c2 = st.columns(2)
    with c1:
        sel_c = st.multiselect(t("p6_cat_label"), safe_cats(df), key="p5c")
    with c2:
        sel_s_display = st.multiselect(t("p6_status_label"),
            [t("status_high"), t("status_warning"), t("status_medium"),
             t("status_lost"), t("status_active")], key="p5s")

    sel_s_int = [s for s in
                 ['High Risk','Warning (Late > 1.5x)','Medium Risk','Lost (Late > 3x)','Active']
                 if t("status_" + {
                     'High Risk':'high','Warning (Late > 1.5x)':'warning',
                     'Medium Risk':'medium','Lost (Late > 3x)':'lost',
                     'Active':'active'}[s]) in sel_s_display] if sel_s_display else []

    dfs = df.copy()
    if sel_c:     dfs = dfs[dfs['product_category_name'].isin(sel_c)]
    if sel_s_int: dfs = dfs[dfs['status'].isin(sel_s_int)]

    agg = {'order_purchase_timestamp': 'count', 'payment_value': 'sum',
           'churn_probability': 'mean', 'delivery_days': 'mean'}
    if 'review_score' in dfs.columns: agg['review_score'] = 'mean'
    ss = dfs.groupby('seller_id').agg(agg).reset_index()
    ss = ss.rename(columns={'order_purchase_timestamp': 'orders'})
    if 'review_score' not in ss.columns: ss['review_score'] = np.nan
    ss = ss[ss['orders'] >= 3]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("p6_metric_shops"), f"{len(ss):,}")
    k2.metric(t("p6_metric_rev"),   f"R$ {ss['payment_value'].sum():,.0f}")
    k3.metric(t("p6_metric_review"),
              f"{ss['review_score'].mean():.2f}" if ss['review_score'].notna().any() else t("p6_na"))
    k4.metric(t("p6_metric_del"),   f"{ss['delivery_days'].mean():.1f}{t('p6_days')}")

    st.markdown("---")
    cs_, cd_ = st.columns([1, 3])
    sort_opts = [t("p6_sort_risk"), t("p6_sort_late"), t("p6_sort_score"),
                 t("p6_sort_rev"), t("p6_sort_vol")]
    with cs_:
        sort_m = st.radio(t("p6_sort"), sort_opts)
    with cd_:
        if sort_m == t("p6_sort_risk"):   sdf = ss.sort_values('churn_probability', ascending=False)
        elif sort_m == t("p6_sort_late"): sdf = ss.sort_values('delivery_days', ascending=False)
        elif sort_m == t("p6_sort_score"):sdf = ss.sort_values('review_score', ascending=True)
        elif sort_m == t("p6_sort_rev"):  sdf = ss.sort_values('payment_value', ascending=False)
        else:                              sdf = ss.sort_values('orders', ascending=False)
        st.dataframe(sdf, column_config={
            "orders": st.column_config.NumberColumn("Orders"),
            "payment_value": st.column_config.NumberColumn("Revenue", format="R$%.0f"),
            "delivery_days": st.column_config.NumberColumn(t("p6_col_del"), format="%.1f"),
            "review_score": st.column_config.NumberColumn("Review", format="%.1f⭐"),
            "churn_probability": st.column_config.ProgressColumn(
                "Risk", format="%.2f", min_value=0, max_value=1)
        }, hide_index=True, use_container_width=True, height=600)

# ==========================================
# PAGE 6 (nav): Buying Cycle
# ==========================================
elif page == t("page_cycle"):
    st.title(t("p4_title"))

    sel_c  = st.multiselect(t("p4_cat_label"), safe_cats(df), key="p6c")
    df_cy  = df[df['product_category_name'].isin(sel_c)].copy() if sel_c else df.copy()

    g_avg  = df['cat_median_days'].mean()
    c_avg  = df_cy['cat_median_days'].mean()
    c_late = df_cy['lateness_score'].mean() if 'lateness_score' in df_cy.columns else 0
    fast   = (df_cy['cat_median_days'] <= 30).sum()

    m1, m2, m3 = st.columns(3)
    m1.metric(t("p4_avg_cycle"), f"{c_avg:.0f}{t('p4_days_unit')}",
              f"{c_avg-g_avg:+.0f}{t('p4_avg_cycle_delta')}", delta_color="inverse")
    m2.metric(t("p4_lateness"), f"{c_late:.2f}x")
    m3.metric(t("p4_fast"), f"{fast:,}{t('p4_people_unit')}")

    st.markdown("---")
    st.subheader(t("p4_trend_title"))
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
                        y=alt.Y('gap', title=t("p4_gap_y"), scale=alt.Scale(zero=False)),
                        color=alt.value('#e67e22'),
                        tooltip=['Date', alt.Tooltip('gap', format='.1f', title=t("p4_gap_tooltip"))]
                    ).properties(height=350), use_container_width=True)
            else:
                st.info(t("p4_no_trend"))
        else:
            st.info(t("p4_no_repeat"))

    st.markdown("---")
    st.subheader(t("p4_detail"))
    summ = df_cy.groupby('product_category_name').agg(
        Customers=('customer_unique_id', 'count'),
        Cycle_Days=('cat_median_days', 'mean'),
        Late_Score=('lateness_score', 'mean'),
        Churn_Risk=('churn_probability', 'mean')
    ).reset_index().sort_values('Cycle_Days')
    st.dataframe(summ, column_config={
        "Customers":  st.column_config.NumberColumn(t("p4_col_cust"), format=f"%d{t('p4_people_unit')}"),
        "Cycle_Days": st.column_config.NumberColumn(t("p4_col_cycle"), format=f"%.0f{t('p4_days_unit')}"),
        "Late_Score": st.column_config.NumberColumn(t("p4_col_late"), format="%.2fx"),
        "Churn_Risk": st.column_config.ProgressColumn("Risk", format="%.2f", min_value=0, max_value=1)
    }, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader(t("p4_heatmap"))
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
                            title=t("p4_heat_month")),
                    y=alt.Y('product_category_name', title=t("p4_heat_cat")),
                    color=alt.Color('vol', scale=alt.Scale(scheme='orangered'),
                                    title=t("p4_heat_vol")),
                    tooltip=['product_category_name', 'month_name',
                             alt.Tooltip('vol', format=',')]
                ).properties(height=500), use_container_width=True)
            st.info(t("p4_heat_tip"))

# ==========================================
# PAGE 7 (nav): Customer Detail
# ==========================================
elif page == t("page_customer"):
    st.title(t("p7_title"))

    status_display_map = {
        'High Risk':           t("status_high"),
        'Warning (Late > 1.5x)': t("status_warning"),
        'Medium Risk':         t("status_medium"),
        'Lost (Late > 3x)':    t("status_lost"),
        'Active':              t("status_active"),
    }
    status_reverse_map = {v: k for k, v in status_display_map.items()}

    with st.expander(t("p7_filters"), expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            all_statuses_display = list(status_display_map.values())
            default_display = [status_display_map['High Risk'],
                               status_display_map['Warning (Late > 1.5x)']]
            sel_status_display = st.multiselect(t("p7_status_label"),
                all_statuses_display, default=default_display)
            sel_status = [status_reverse_map[s] for s in sel_status_display
                          if s in status_reverse_map]
        with f2:
            sel_cats = st.multiselect(t("p7_cat_label"), safe_cats(df))
        with f3:
            search_id = st.text_input(t("p7_search"), "")

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
            st.subheader(t("p7_top10"))
            base   = alt.Chart(cat_s.head(10)).encode(
                y=alt.Y('product_category_name', sort='-x', title=None))
            b_tot  = base.mark_bar(color='#f0f2f6').encode(x='Total')
            b_risk = base.mark_bar(color='#e74c3c').encode(x='Risk')
            st.altair_chart(b_tot + b_risk, use_container_width=True)
        with ct:
            st.subheader(t("p7_detail"))
            st.dataframe(cat_s, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(t("p7_list", n=f"{len(filtered):,}"))
    show = [c for c in ['customer_unique_id', 'status', 'churn_probability',
                        'lateness_score', 'cat_median_days', 'payment_value',
                        'product_category_name'] if c in df.columns]
    st.dataframe(
        filtered[show].sort_values('churn_probability', ascending=False),
        column_config={
            "churn_probability": st.column_config.ProgressColumn(
                "Risk", format="%.2f", min_value=0, max_value=1),
            "lateness_score": st.column_config.NumberColumn(
                t("p7_col_late"), format="%.1fx")
        }, use_container_width=True)
