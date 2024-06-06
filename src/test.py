banned = [1, 6, 5]
n = 5
max_sum = 6

# 1->n
# sum <= max_sum
# max_count

arr = []
k = 3
# min len sub arr with k distinct numbers
# n < 10^5

# left_set # 0 -> left - 1
# left_pointer

# right_set # left - right
# right_poiter


arr = [2, 3, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
k = 5

# left = 0 -> 8
# left = 1 -> 8
# 2 -> 8
# 3 -> 8
# 4 -> 8
# 5 -> 8

# counter = {2: 0, 3:1, 4, 5, 6, }

# right_set        1
# right_poiter     1
# left_set
# left_pointer     0

from collections import defaultdict

def min_sub_arr(arr: list, k: int):
    # left = 0
    right = 0
    counter = defaultdict(lambda: 0)

    min_len = len(arr)
    for left in range(arr):
        
        if left > 0 and counter[arr[left - 1]] == 1:
            del counter[arr[left - 1]]
        else:
            counter[arr[left - 1]] = counter[arr[left - 1]] - 1

        while len(counter) < k:
            if right == n:
                return min_len
            right += 1
            counter[arr[right]] = counter.get(arr[right], 0) + 1

        min_len = min(right - left + 1, min_len)

    return min_len


# shop date, shop_id, revenue -> key: date, shop_id
# top 3 date revenue

SELECT shop_id, date, revenue
FROM (
    SELECT date, shop_id, revenue,
        ROW_NUMBER() OVER(PARTITION BY shop_id ORDER BY revenue DESC) AS RN
    FROM shop
) AS A
WHERE RN <= 3
ORDER BY shop_id, date ASC

