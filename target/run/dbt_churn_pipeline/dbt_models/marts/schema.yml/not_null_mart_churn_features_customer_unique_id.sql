
    
    select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
  
    
    



select customer_unique_id
from `academic-moon-483615-t2`.`analytics_olist`.`mart_churn_features`
where customer_unique_id is null



  
  
      
    ) dbt_internal_test