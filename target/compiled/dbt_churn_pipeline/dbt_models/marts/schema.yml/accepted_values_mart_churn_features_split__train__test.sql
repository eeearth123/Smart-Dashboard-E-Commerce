
    
    

with all_values as (

    select
        split as value_field,
        count(*) as n_records

    from `academic-moon-483615-t2`.`analytics_olist`.`mart_churn_features`
    group by split

)

select *
from all_values
where value_field not in (
    'train','test'
)


