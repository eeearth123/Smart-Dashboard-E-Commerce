select
    order_id,
    cast(review_score as int64) as review_score
from {{ source('raw_olist', 'raw_order_reviews') }}
