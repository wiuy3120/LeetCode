SELECT 
    IFNULL((
            SELECT
                salary
            FROM Employee
            ORDER BY salary DESC
            LIMIT 1, 1
        ),
        NULL
    ) AS SecondHighestSalary;

