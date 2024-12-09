# pyright: reportRedeclaration=false
import bisect
import math
import random
from collections import Counter, deque
from heapq import heapify, heappop, heappush, heappushpop, heapreplace, nlargest
from itertools import accumulate
from typing import Dict, List, Optional, Set, Tuple

from sortedcontainers import SortedList

# Binary Search


# 2070. Most Beautiful Item for Each Query
class Solution:
    def maximumBeauty(
        self, items: List[List[int]], queries: List[int]
    ) -> List[int]:
        from bisect import bisect

        sorted_items = [[0, 0]] + sorted(items)

        for i, item in enumerate(sorted_items[1:]):
            item[1] = max(item[1], sorted_items[i][1])

        return [
            sorted_items[bisect(sorted_items, [query + 1]) - 1][1]
            for query in queries
        ]


# 2461. Maximum Sum of Distinct Subarrays With Length K
class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        counter = Counter(nums[:k])
        distinct_count = len(counter)
        cur_sum = sum(nums[:k])
        res = 0
        if distinct_count == k:
            res = cur_sum

        for i, num in enumerate(nums[:-k]):
            counter[num] -= 1
            if counter[num] == 0:
                distinct_count -= 1

            next_num = nums[i + k]
            if counter[next_num] == 0:
                distinct_count += 1
                counter[next_num] = 1
            else:
                counter[next_num] += 1

            cur_sum += next_num - num
            if distinct_count == k:
                res = max(res, cur_sum)

        return res

    # 2109. Adding Spaces to a String
    def addSpaces(self, s: str, spaces: List[int]) -> str:
        word_list = [s[: spaces[0]]]
        for i in range(len(spaces) - 1):
            word_list.append(s[spaces[i] : spaces[i + 1]])
        word_list.append(s[spaces[-1] :])

        return " ".join(word_list)

    # 2825. Make String a Subsequence Using Cyclic Increments
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        def increase_chart(chart: str):
            if chart == "z":
                return "a"
            return chr(ord(chart) + 1)

        if len(str1) < len(str2):
            return False

        n = len(str2)
        idx = 0
        for c in str1:
            if c == str2[idx] or increase_chart(c) == str2[idx]:
                idx += 1
                if idx == n:
                    return True

        return False

    # 2337. Move Pieces to Obtain a String
    def canChange(self, start: str, target: str) -> bool:
        start_indices = [
            i + 1 if c == "L" else -i - 1
            for i, c in enumerate(start)
            if c in ["L", "R"]
        ]
        target_indices = [
            i + 1 if c == "L" else -i - 1
            for i, c in enumerate(target)
            if c in ["L", "R"]
        ]

        if len(start_indices) != len(target_indices):
            return False

        for idx1, idx2 in zip(start_indices, target_indices):
            if idx1 * idx2 < 0 or idx1 < idx2:
                return False

        return True

    # 1760. Minimum Limit of Balls in a Bag
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        heap = [(-num, -num, 1) for num in nums]
        heapify(heap)
        for _ in range(maxOperations):
            max_item = heap[0]
            num, divisor = -max_item[1], max_item[2]
            heappushpop(
                heap, (-math.ceil(num / (divisor + 1)), -num, divisor + 1)
            )
        return -heap[0][0]

    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        left, right = 0, max(nums)

        while left < right:
            mid = (left + right) // 2
            num_ops = sum((num - 1) // mid for num in nums)

            if num_ops > maxOperations:
                left = mid + 1
            else:
                right = mid

        return left

    # 2054. Two Best Non-Overlapping Events
    def maxTwoEvents(self, events: List[List[int]]) -> int:

        def acc_max_func(last: Tuple[int, int], current: Tuple[int, int]):
            return (
                current[0],
                max(last[1], current[1]),
            )

        start_times = list(
            accumulate(
                sorted(
                    [(event[0], event[2]) for event in events], reverse=True
                ),
                acc_max_func,
            )
        )[::-1]
        print(start_times)
        end_times = list(
            accumulate(
                sorted([(event[1], event[2]) for event in events]),
                acc_max_func,
            )
        )
        print(end_times)
        max_sum = 0
        for end_time, value in end_times:
            idx = bisect.bisect(start_times, end_time, key=lambda x: x[0])
            if idx == len(start_times):
                max_sum = max(max_sum, end_times[-1][1])
                break
            max_sum = max(max_sum, value + start_times[idx][1])

        return max_sum

    # 3152. Special Array II
    def isArraySpecial(
        self, nums: List[int], queries: List[List[int]]
    ) -> List[bool]:
        boundary_indices = [
            i for i in range(len(nums) - 1) if (nums[i] + nums[i + 1]) % 2 == 0
        ] + [len(nums) - 1]

        def is_special(query: Tuple[int, int]):
            start_idx = bisect.bisect_left(boundary_indices, query[0])
            boundary = boundary_indices[start_idx]
            return query[1] <= boundary

        return [is_special(query) for query in queries]

    def isArraySpecial(
        self, nums: List[int], queries: List[List[int]]
    ) -> List[bool]:
        part_counter = 0
        parts = [0]

        for i, num in enumerate(nums[1:]):
            if (num + nums[i]) % 2 == 0:
                part_counter += 1
            parts.append(part_counter)

        return [parts[start] == parts[end] for start, end in queries]
