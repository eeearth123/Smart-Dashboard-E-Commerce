select
    order_id,
    cast(review_score as int64) as review_score
from `academic-moon-483615-t2`.`raw_olist`.`raw_order_reviews`