with customers as (
    select * from {{ ref('stg_customers') }}
),
orders as (
    select * from {{ ref('stg_orders') }}
),
payments as (
    -- รวมข้อมูลการชำระเงินรายคำสั่งซื้อ (บางคำสั่งซื้อจ่ายหลายรอบ/หลายประเภท)
    select
        order_id,
        max(payment_sequential) as payment_sequential,
        max(payment_installments) as payment_installments,
        sum(payment_value) as payment_value,
        max(case when payment_type = 'voucher' then 1 else 0 end) as uses_voucher
    from {{ ref('stg_order_payments') }}
    group by 1
),
items as (
    -- รวมข้อมูลราคาและค่าจัดส่งสินค้าในแต่ละคำสั่งซื้อ
    select
        order_id,
        sum(price) as price,
        sum(freight_value) as freight_value,
        max(product_id) as product_id,
        max(seller_id) as seller_id
    from {{ ref('stg_order_items') }}
    group by 1
),
reviews as (
    -- หาคะแนนรีวิวเฉลี่ยในแต่ละคำสั่งซื้อ
    select
        order_id,
        avg(review_score) as review_score
    from {{ ref('stg_order_reviews') }}
    group by 1
),
products as (
    select * from {{ ref('stg_products') }}
)

select
    o.order_id,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    coalesce(i.price, 0) as price,
    coalesce(i.freight_value, 0) as freight_value,
    coalesce(p.payment_installments, 1) as payment_installments,
    coalesce(p.payment_value, 0) as payment_value,
    coalesce(p.payment_sequential, 1) as payment_sequential,
    coalesce(p.uses_voucher, 0) as uses_voucher,
    coalesce(r.review_score, 3) as review_score,
    pr.product_category_name,
    i.seller_id,
    -- คำนวณวันที่ซื้อล่าสุดเทียบกับวันสุดท้ายของระบบ:
    -- หากเกิน 180 วัน -> เป็นข้อมูลมีผลเฉลยแล้ว นำไปใช้เทรน (train)
    -- หากยังไม่เกิน 180 วัน -> เป็นลูกค้าปัจจุบันที่ต้องทำนายผลและโชว์บน Dashboard (test)
    case 
        when timestamp_diff(
            max(o.order_purchase_timestamp) over (),
            max(o.order_purchase_timestamp) over (partition by c.customer_unique_id),
            DAY
        ) >= 180 then 'train'
        else 'test'
    end as split
from orders o
join customers c on o.customer_id = c.customer_id
left join items i on o.order_id = i.order_id
left join payments p on o.order_id = p.order_id
left join reviews r on o.order_id = r.order_id
left join products pr on i.product_id = pr.product_id
