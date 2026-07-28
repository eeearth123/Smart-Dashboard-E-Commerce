select
    product_id,
    case
        when product_category_name in (
            'bed_bath_table', 'furniture_decor', 'housewares', 'furniture_living_room',
            'furniture_bedroom', 'furniture_mattress_and_upholstery', 'kitchen_dining_laundry_garden_furniture',
            'office_furniture', 'garden_tools', 'home_confort', 'home_comfort_2', 'air_conditioning', 'flowers', 'la_cuisine'
        ) then 'Home & Furniture'
        
        when product_category_name in (
            'computers_accessories', 'telephony', 'electronics', 'computers', 'tablets_printing_image',
            'fixed_telephony', 'signaling_and_security', 'security_and_services', 'audio'
        ) then 'Electronics & Tech'
        
        when product_category_name in (
            'home_appliances', 'home_appliances_2', 'small_appliances', 'small_appliances_home_oven_and_coffee'
        ) then 'Appliances'
        
        when product_category_name in (
            'construction_tools_construction', 'construction_tools_lights', 'construction_tools_safety',
            'costruction_tools_garden', 'costruction_tools_tools', 'home_construction'
        ) then 'Construction & Tools'
        
        when product_category_name in (
            'sports_leisure', 'fashion_sport'
        ) then 'Sports & Leisure'
        
        when product_category_name in (
            'health_beauty', 'perfumery', 'diapers_and_hygiene'
        ) then 'Health & Beauty'
        
        when product_category_name in (
            'watches_gifts', 'fashion_bags_accessories', 'fashion_shoes', 'fashion_male_clothing',
            'fashion_underwear_beach', 'fashio_female_clothing', 'fashion_childrens_clothes', 'luggage_accessories'
        ) then 'Fashion & Accessories'
        
        when product_category_name in (
            'toys', 'baby', 'consoles_games', 'cool_stuff'
        ) then 'Toys & Games'
        
        when product_category_name in (
            'books_general_interest', 'books_technical', 'books_imported', 'stationery',
            'dvds_blu_ray', 'musical_instruments', 'music', 'cds_dvds_musicals', 'art',
            'arts_and_craftmanship', 'cine_photo', 'party_supplies'
        ) then 'Books, Art & Media'
        
        when product_category_name = 'auto' then 'Auto'
        
        when product_category_name in (
            'food', 'drinks', 'food_drink'
        ) then 'Food & Drinks'
        
        when product_category_name in (
            'industry_commerce_and_business', 'agro_industry_and_commerce', 'market_place'
        ) then 'Industry & Business'
        
        else 'Others'
    end as product_category_name
from `academic-moon-483615-t2`.`raw_olist`.`raw_products`