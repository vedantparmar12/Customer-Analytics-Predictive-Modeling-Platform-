# Olist E-commerce Dataset Documentation

## Overview
This dataset contains information about orders made at Olist Store, a Brazilian e-commerce marketplace that connects small businesses to large marketplaces. The dataset has multiple CSV files containing information about customers, products, orders, payments, reviews, and more, spanning from 2016 to 2018.

## Dataset Files Description

### 1. olist_customers_dataset.csv (99,441 records)
Contains customer information and their location.

**Columns:**
- `customer_id`: Unique customer identifier
- `customer_unique_id`: Unique identifier of a customer (anonymized)
- `customer_zip_code_prefix`: First 5 digits of customer zip code
- `customer_city`: Customer city name
- `customer_state`: Customer state (Brazilian state code)

### 2. olist_geolocation_dataset.csv (1,000,163 records)
Brazilian zip codes and their lat/lng coordinates.

**Columns:**
- `geolocation_zip_code_prefix`: First 5 digits of zip code
- `geolocation_lat`: Latitude
- `geolocation_lng`: Longitude
- `geolocation_city`: City name
- `geolocation_state`: State code

### 3. olist_order_items_dataset.csv (112,650 records)
Contains data about items purchased within each order.

**Columns:**
- `order_id`: Unique identifier of the order
- `order_item_id`: Sequential number identifying number of items included in the same order
- `product_id`: Unique product identifier
- `seller_id`: Unique seller identifier
- `shipping_limit_date`: Seller shipping limit date for handling order
- `price`: Item price
- `freight_value`: Freight value (shipping cost)

### 4. olist_order_payments_dataset.csv (103,886 records)
Contains order payment information.

**Columns:**
- `order_id`: Unique identifier of an order
- `payment_sequential`: Payment sequence for orders with multiple payment methods
- `payment_type`: Method of payment (credit_card, boleto, voucher, debit_card)
- `payment_installments`: Number of payment installments
- `payment_value`: Transaction value

### 5. olist_order_reviews_dataset.csv (100,000 records)
Contains customer reviews and ratings.

**Columns:**
- `review_id`: Unique review identifier
- `order_id`: Unique order identifier
- `review_score`: Rating from 1 to 5
- `review_comment_title`: Review title (optional)
- `review_comment_message`: Review message (optional)
- `review_creation_date`: Date review was created
- `review_answer_timestamp`: Date review was answered

### 6. olist_orders_dataset.csv (99,441 records)
Core dataset containing order information.

**Columns:**
- `order_id`: Unique order identifier
- `customer_id`: Customer identifier
- `order_status`: Order status (delivered, shipped, canceled, etc.)
- `order_purchase_timestamp`: Purchase timestamp
- `order_approved_at`: Payment approval timestamp
- `order_delivered_carrier_date`: Order posting date to carrier
- `order_delivered_customer_date`: Actual delivery date
- `order_estimated_delivery_date`: Estimated delivery date

### 7. olist_products_dataset.csv (32,951 records)
Contains product information.

**Columns:**
- `product_id`: Unique product identifier
- `product_category_name`: Product category (in Portuguese)
- `product_name_lenght`: Number of characters in product name
- `product_description_lenght`: Number of characters in product description
- `product_photos_qty`: Number of product photos
- `product_weight_g`: Product weight in grams
- `product_length_cm`: Product length in cm
- `product_height_cm`: Product height in cm
- `product_width_cm`: Product width in cm

### 8. olist_sellers_dataset.csv (3,095 records)
Contains seller information.

**Columns:**
- `seller_id`: Unique seller identifier
- `seller_zip_code_prefix`: First 5 digits of seller zip code
- `seller_city`: Seller city name
- `seller_state`: Seller state

### 9. product_category_name_translation.csv (71 records)
Translation of product categories from Portuguese to English.

**Columns:**
- `product_category_name`: Category name in Portuguese
- `product_category_name_english`: Category name in English

## Data Relationships

The datasets are interconnected through various ID fields:

1. **Orders** are the central entity, connected to:
   - Customers via `customer_id`
   - Order items via `order_id`
   - Payments via `order_id`
   - Reviews via `order_id`

2. **Order Items** connect:
   - Orders to Products via `product_id`
   - Orders to Sellers via `seller_id`

3. **Geographic Data**:
   - Customers and Sellers can be mapped to geolocation data via `zip_code_prefix`

4. **Product Categories**:
   - Products have categories that can be translated to English

## Key Business Metrics Available

1. **Sales Performance**: Revenue, order volume, average order value
2. **Customer Behavior**: Purchase patterns, review ratings, geographic distribution
3. **Product Analytics**: Best-selling products, category performance
4. **Seller Performance**: Sales by seller, geographic distribution
5. **Delivery Performance**: Delivery times vs estimates
6. **Payment Methods**: Payment type preferences and installment patterns

## Data Quality Notes

- The dataset is anonymized to protect customer and seller privacy
- Some fields may contain null values (e.g., review comments are optional)
- Dates are in UTC format
- Monetary values are in Brazilian Real (BRL)
- Geographic data uses Brazilian zip code system

## Use Cases

This dataset is ideal for:
- Customer segmentation and RFM analysis
- Sales forecasting and trend analysis
- Product recommendation systems
- Delivery time optimization
- Seller performance evaluation
- Customer sentiment analysis from reviews
- Geographic market analysis
- Payment behavior studies