

  create or replace view `academic-moon-483615-t2`.`analytics_olist`.`stg_customers`
  OPTIONS()
  as select
    customer_id,
    customer_unique_id
from `academic-moon-483615-t2`.`raw_olist`.`raw_customers`;

