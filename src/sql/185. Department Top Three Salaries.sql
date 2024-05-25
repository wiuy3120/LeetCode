SELECT
    B.name AS Department,
    A.name AS Employee,
    A.salary AS Salary
FROM (
    SELECT
        departmentId,
        name,
        salary,
        DENSE_RANK() OVER(PARTITION BY departmentId ORDER BY salary DESC) AS `rank`
    FROM Employee
) AS A
LEFT JOIN Department AS B
    ON A.departmentId = B.id
WHERE A.`rank` <= 3;

WITH CTE AS (
    SELECT
        departmentId,
        name,
        salary,
        DENSE_RANK() OVER(PARTITION BY departmentId ORDER BY salary DESC) AS `rank`
    FROM Employee
)
SELECT
    B.name AS Department,
    A.name AS Employee,
    A.salary AS Salary
FROM CTE AS A
LEFT JOIN Department AS B
    ON A.departmentId = B.id
WHERE A.`rank` <= 3;



WITH CTE AS (
    SELECT
        departmentId,
        A.name AS name,
        salary,
        B.name AS dept_name,
        DENSE_RANK() OVER(PARTITION BY departmentId ORDER BY salary DESC) AS `rank`
    FROM Employee AS A
    LEFT JOIN Department AS B
    ON A.departmentId = B.id
)
SELECT
    dept_name AS Department,
    name AS Employee,
    salary AS Salary
FROM CTE
WHERE `rank` <= 3;