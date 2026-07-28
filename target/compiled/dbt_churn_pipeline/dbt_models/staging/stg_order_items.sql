select
    order_id,
    product_id,
    cast(price as numeric) as price,
    cast(freight_value as numeric) as freight_value
from `academic-moon-483615-t2`.`raw_olist`.`raw_order_items`