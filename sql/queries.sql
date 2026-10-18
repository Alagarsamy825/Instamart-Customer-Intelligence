-- Each query starts with a "-- name: Qn" line so run_queries.py can split the file.
-- Only completed orders count as revenue or spend.

-- name: Q1
-- WHERE filters rows first, HAVING filters the grouped totals afterwards.
SELECT customer_id, SUM(order_amount) AS total_amount FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_amount DESC;

-- name: Q2
-- Revenue by city and product category.
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN products AS p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- name: Q3
-- Products priced above the catalogue average.
SELECT
    product_id,
    category,
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- name: Q4
-- The completed-order condition lives in the ON clause so customers without
-- one still show up; they are the rows where the order side is NULL.
SELECT
    c.customer_id,
    c.city
FROM customers AS c
LEFT JOIN orders AS o
       ON o.customer_id = c.customer_id
      AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- name: Q5
WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS avg_spend
    FROM customer_spend
)
SELECT
    cs.customer_id,
    ROUND(cs.total_spend, 2) AS total_spend,
    ROUND(a.avg_spend, 2)    AS avg_spend
FROM customer_spend AS cs
CROSS JOIN average_spend AS a
WHERE cs.total_spend > a.avg_spend
ORDER BY cs.total_spend DESC;

-- name: Q6
WITH customer_spend AS (
    SELECT
        o.customer_id,
        c.city,
        ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders AS o
    JOIN customers AS c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
)
SELECT
    city,
    customer_id,
    total_spend,
    RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_dense_rank,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_row_number
FROM customer_spend
ORDER BY city, spend_row_number;

-- name: Q7
WITH daily AS (
    SELECT
        SUBSTR(order_ts, 1, 10)       AS day,
        ROUND(SUM(order_amount), 2)   AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    day,
    revenue,
    ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily
ORDER BY day;