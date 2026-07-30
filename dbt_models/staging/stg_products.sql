with products as (
    select * from {{ source('raw_olist', 'raw_products') }}
),
translations as (
    select 
        string_field_0 as pt_name,
        string_field_1 as en_name
    from {{ source('raw_olist', 'product_category_name_translation') }}
    where string_field_0 != 'product_category_name'
)
select
    p.product_id,
    case
        when coalesce(t.en_name, p.product_category_name) in (
            'bed_bath_table', 'furniture_decor', 'housewares', 'furniture_living_room',
            'furniture_bedroom', 'furniture_mattress_and_upholstery', 'kitchen_dining_laundry_garden_furniture',
            'office_furniture', 'garden_tools', 'home_confort', 'home_comfort_2', 'air_conditioning', 'flowers', 'la_cuisine'
        ) then 'Home & Furniture'
        
        when coalesce(t.en_name, p.product_category_name) in (
            'computers_accessories', 'telephony', 'electronics', 'computers', 'tablets_printing_image',
            'fixed_telephony', 'signaling_and_security', 'security_and_services', 'audio'
        ) then 'Electronics & Tech'
        
        when coalesce(t.en_name, p.product_category_name) in (
            'home_appliances', 'home_appliances_2', 'small_appliances', 'small_appliances_home_oven_and_coffee'
        ) then 'Appliances'
        
        when coalesce(t.en_name, p.product_category_name) in (
            'construction_tools_construction', 'construction_tools_lights', 'construction_tools_safety',
            'costruction_tools_garden', 'costruction_tools_tools', 'home_construction'
        ) then 'Construction & Tools'
        
        when coalesce(t.en_name, p.product_category_name) in (
            'sports_leisure', 'fashion_sport'
        ) then 'Sports & Leisure'
        
        when coalesce(t.en_name, p.product_category_name) in (
            'health_beauty', 'perfumery', 'diapers_and_hygiene'
        ) then 'Health & Beauty'
        
        when coalesce(t.en_name, p.product_category_name) in (
            'watches_gifts', 'fashion_bags_accessories', 'fashion_shoes', 'fashion_male_clothing',
            'fashion_underwear_beach', 'fashio_female_clothing', 'fashion_childrens_clothes', 'luggage_accessories'
        ) then 'Fashion & Accessories'
        
        when coalesce(t.en_name, p.product_category_name) in (
            'toys', 'baby', 'consoles_games', 'cool_stuff'
        ) then 'Toys & Games'
        
        when coalesce(t.en_name, p.product_category_name) in (
            'books_general_interest', 'books_technical', 'books_imported', 'stationery',
            'dvds_blu_ray', 'musical_instruments', 'music', 'cds_dvds_musicals', 'art',
            'arts_and_craftmanship', 'cine_photo', 'party_supplies'
        ) then 'Books, Art & Media'
        
        when coalesce(t.en_name, p.product_category_name) = 'auto' then 'Auto'
        
        when coalesce(t.en_name, p.product_category_name) in (
            'food', 'drinks', 'food_drink'
        ) then 'Food & Drinks'
        
        when coalesce(t.en_name, p.product_category_name) in (
            'industry_commerce_and_business', 'agro_industry_and_commerce', 'market_place'
        ) then 'Industry & Business'
        
        else 'Others'
    end as product_category_name
from products p
left join translations t on p.product_category_name = t.pt_name
