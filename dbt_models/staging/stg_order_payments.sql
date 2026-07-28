select
    order_id,
    cast(payment_sequential as int64) as payment_sequential,
    payment_type,
    cast(payment_installments as int64) as payment_installments,
    cast(payment_value as numeric) as payment_value
from {{ source('raw_olist', 'raw_order_payments') }}
