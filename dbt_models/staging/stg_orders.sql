select
    order_id,
    customer_id,
    order_status,
    cast(order_purchase_timestamp as timestamp) as order_purchase_timestamp,
    cast(order_delivered_customer_date as timestamp) as order_delivered_customer_date,
    cast(order_estimated_delivery_date as timestamp) as order_estimated_delivery_date
from {{ source('raw_olist', 'raw_orders') }}
