-- 595. Big Countries
SELECT name, population, area
FROM World
WHERE area >= 3000000 OR population >= 25000000

-- 1757. Recyclable and Low Fat Products
SELECT product_id
FROM Products
WHERE low_fats='Y' AND recyclable='Y';

-- 584. Find Customer Referee
SELECT  name
FROM Customer
WHERE referee_id != 2 OR referee_id IS NULL;

-- 183. Customers Who Never Order
SELECT Customers.name AS Customers
FROM Customers
LEFT JOIN Orders
ON Orders.customerId = Customers.id
WHERE Orders.customerId IS NULL;

-- 1873. Calculate Special Bonus
SELECT employee_id, 
CASE 
    WHEN MOD(employee_id,2) = 1 AND name NOT LIKE ('M%') THEN salary
    ELSE 0
    END AS bonus
FROM Employees
ORDER BY employee_id;

-- 627. Swap Salary
UPDATE Salary
SET sex = 
    CASE WHEN sex = 'f' THEN 'm' 
        ELSE 'f' 
    END
WHERE sex IN ('m', 'f');

-- 196. Delete Duplicate Emails
DELETE p1
FROM Person p1, Person p2
WHERE p1.Email = p2.Email AND p1.Id > p2.Id

-- 1667. Fix Names in a Table
SELECT user_id,  CONCAT(UPPER(LEFT(name, 1)), (LCASE(SUBSTRING(name, 2)))) AS name
FROM  users
ORDER BY user_id

-- 1484. Group Sold Products By The Date
SELECT sell_date, 
    COUNT(DISTINCT(product)) AS num_sold,
    GROUP_CONCAT(DISTINCT(product)) as products
FROM Activities
GROUP BY sell_date
ORDER BY sell_date, product;

-- 1527. Patients With a Condition
SELECT patient_id, patient_name, conditions
FROM Patients
WHERE conditions LIKE ('% DIAB1%');

-- 608. Tree Node
SELECT id, 
    CASE
        WHEN p_id IS NULL THEN 'Root'
        WHEN id IN (
            SELECT DISTINCT p_id 
            FROM Tree
        ) THEN 'Inner' # If ID is a parent node and have child nodes
        ELSE 'Leaf'
        END AS type
FROM Tree;

-- 176. Second Highest Salary
SELECT
    CASE  
        WHEN COUNT(salary) = 1 THEN NULL
        ELSE 
            (SELECT MAX(salary)
            FROM Employee
            WHERE salary <(
                SELECT MAX(salary)
                FROM Employee
                )
            )
    END as SecondHighestSalary
FROM Employee;

-- 175. Combine Two Tables
SELECT firstName, lastName, city, state
FROM Person
LEFT JOIN Address
ON  Person.personId = Address.personId;

-- 1581. Customer Who Visited but Did Not Make Any Transactions
SELECT customer_id, COUNT(visit_id) AS count_no_trans
FROM(
    SELECT v.customer_id, v.visit_id
    FROM Visits AS v
    LEFT JOIN Transactions AS t
    ON v.visit_id = t.visit_id 
    WHERE t.visit_id IS NULL
) AS joined
GROUP BY customer_id
ORDER BY count_no_trans;

-- 1587. Bank Account Summary II
SELECT name, SUM(amount) as balance
FROM Users as u
LEFT JOIN Transactions as t
ON u.account = t.account
GROUP BY u.account
HAVING balance > 10000;

-- 1148. Article Views I
SELECT DISTINCT author_id as id
FROM Views
WHERE author_id = viewer_id
ORDER BY ID;

-- 197. Rising Temperature
SELECT w1.id
FROM Weather AS w1, Weather AS w2
WHERE w1.recordDate = DATE_ADD(w2.recordDate, INTERVAL 1 DAY)
AND w1.temperature > w2.temperature;

-- 607. Sales Person
SELECT name
FROM SalesPerson 
WHERE sales_id NOT IN (SELECT sales_id
                        FROM Orders AS o
                        JOIN Company AS c
                        ON o.com_id = c.com_id
                        WHERE name = 'RED');

-- 1141. User Activity for the Past 30 Days I
SELECT activity_date as day, COUNT(DISTINCT(user_id)) AS active_users
FROM Activity
WHERE activity_date BETWEEN DATE_SUB('2019-07-27', INTERVAL 29 day) AND '2019-07-27'
GROUP BY activity_date
ORDER BY activity_date;

-- 1693. Daily Leads and Partners
SELECT date_id, make_name, COUNT(DISTINCT(lead_id)) AS unique_leads, 
                            COUNT(DISTINCT(partner_id)) AS unique_partners
FROM DailySales
GROUP BY date_id, make_name;

-- 1729. Find Followers Count
SELECT user_id, COUNT(follower_id) AS followers_count
FROM Followers
GROUP BY user_id
ORDER BY user_id;

-- 586. Customer Placing the Largest Number of Orders
SELECT customer_number
FROM Orders
WHERE order_number = (
  SELECT MAX(order_number)
   FROM Orders
  );

-- 586. Customer Placing the Largest Number of Orders
SELECT customer_number
FROM Orders
GROUP BY customer_number
HAVING COUNT(order_number) = (
  SELECT MAX(mycount)
  FROM (
    SELECT customer_number, COUNT(order_number) AS mycount
    FROM Orders
    GROUP BY customer_number
  ) AS innercount
);

    -- select
    --     customer_number
    -- from orders
    -- group by customer_number
    -- order by count(*) desc
    -- limit 1;

-- 511. Game Play Analysis I
SELECT player_id, MIN(event_date) AS first_login
FROM Activity
GROUP BY player_id;

-- 1890. The Latest Login in 2020
SELECT user_id, MAX(time_stamp) AS last_stamp
FROM Logins
WHERE YEAR(time_stamp) = '2020'
GROUP BY user_id;

-- 1741. Find Total Time Spent by Each Employee
SELECT event_day AS day, emp_id, SUM(out_time - in_time) AS total_time
FROM Employees
GROUP BY event_day, emp_id

-- 1393. Capital Gain/Loss
SELECT stock_name,
    SUM(IF(operation = 'Sell', price, -price)) AS capital_gain_loss
FROM Stocks 
GROUP BY stock_name;

-- 1407. Top Travellers
SELECT name, 
  IF ( r.user_id IS NULL, 0, SUM(distance)) AS travelled_distance
FROM Users AS u
LEFT JOIN Rides AS r
ON u.id = r.user_id
GROUP BY u.id
ORDER BY travelled_distance DESC, name ASC

-- 1158. Market Analysis I
SELECT user_id AS buyer_id, join_date, COUNT(order_id) AS orders_in_2019
FROM Users AS u
LEFT JOIN Orders AS o
ON u.user_id = o.buyer_id AND YEAR(order_date) = '2019'
GROUP BY user_id

-- 182. Duplicate Emails
SELECT email AS Email
FROM Person
GROUP BY email
HAVING COUNT(email) > 1;

-- 1050. Actors and Directors Who Cooperated At Least Three Times
SELECT actor_id, director_id
FROM ActorDirector
GROUP BY actor_id, director_id
HAVING COUNT(*) >= 3;

-- 1084. Sales Analysis III
SELECT p.product_id, p.product_name
FROM Product AS p
JOIN Sales AS s
ON s.product_id = p.product_id
GROUP BY p.product_id
HAVING  MIN(s.sale_date) >= '2019-01-01' AND  MAX(s.sale_date) <= '2019-03-31';