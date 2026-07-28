
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select order_id
from `academic-moon-483615-t2`.`analytics_olist`.`stg_orders`
where order_id is null



  
  
      
    ) dbt_internal_test