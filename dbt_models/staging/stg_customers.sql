select
    customer_id,
    customer_unique_id
from {{ source('raw_olist', 'raw_customers') }}
