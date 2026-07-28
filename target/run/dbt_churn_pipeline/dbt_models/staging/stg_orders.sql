

  create or replace view `academic-moon-483615-t2`.`analytics_olist`.`stg_orders`
  OPTIONS()
  as select
    order_id,
    customer_id,
    order_status,
    cast(order_purchase_timestamp as timestamp) as order_purchase_timestamp,
    cast(order_delivered_customer_date as timestamp) as order_delivered_customer_date,
    cast(order_estimated_delivery_date as timestamp) as order_estimated_delivery_date
from `academic-moon-483615-t2`.`raw_olist`.`raw_orders`;

