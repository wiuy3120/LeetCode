SELECT DISTINCT num AS ConsecutiveNums 
FROM (
    SELECT
        num,
        LAG(num) OVER w AS last_num,
        LEAD(num) OVER w aS next_num,
        id,
        LAG(id) OVER w AS last_id,
        LEAD(id) OVER w aS next_id
    FROM Logs
    WINDOW w AS (ORDER BY id)
) AS A
WHERE num = last_num
    AND num = next_num
    AND id = last_id + 1
    AND id = next_id - 1;

SELECT DISTINCT num AS ConsecutiveNums 
FROM (
    SELECT
        num,
        LAG(num) OVER w AS lag_num,
        LAG(num, 2) OVER w aS lag2_num,
        id,
        LAG(id) OVER w AS lag_id,
        LAG(id, 2) OVER w aS lag2_id
    FROM Logs
    WINDOW w AS (ORDER BY id)
) AS A
WHERE num = lag_num
    AND num = lag2_num
    AND id = lag_id + 1
    AND id = lag2_id + 2;

SELECT DISTINCT A.num AS ConsecutiveNums 
FROM Logs A 
JOIN Logs B ON A.id = B.id - 1
JOIN Logs C ON A.id = C.id - 2
WHERE A.num = B.num AND A.num = C.num