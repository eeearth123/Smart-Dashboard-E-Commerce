select
    customer_id,
    customer_unique_id,
    customer_city,
    customer_state
from {{ source('raw_olist', 'raw_customers') }}
