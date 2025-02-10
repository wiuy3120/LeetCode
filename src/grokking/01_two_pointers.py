# pyright: reportRedeclaration=false
import bisect
import math
import operator
import random
from collections import Counter, deque
from heapq import heapify, heappop, heappush, heappushpop, heapreplace, nlargest
from itertools import accumulate
from typing import Dict, List, Optional, Set, Tuple

from sortedcontainers import SortedList


class Solution:
    # Pair with Target Sum
    # 167. Two Sum II - Input Array Is Sorted
    def search(self, arr, target_sum):
        left, right = 0, len(arr) - 1
        while left < right:
            two_sum = arr[left] + arr[right]
            if two_sum == target_sum:
                return [left, right]
            elif two_sum < target_sum:
                left += 1
            else:
                right += 1

        return [-1, -1]

    # Find Non-Duplicate Number Instances
    # 26. Remove Duplicates from Sorted Array
    def removeDuplicates(self, nums: List[int]) -> int:
        nunique = 0
        for num in nums[1:]:
            if num != nums[nunique]:
                nunique += 1
                nums[nunique] = num
        return nunique + 1

    # 977. Squares of a Sorted Array
    def sortedSquares(self, nums: List[int]) -> List[int]:
        res = [0] * len(nums)
        left, right = 0, len(nums) - 1

        while left < right:
            if nums[right] ** 2 > nums[left] ** 2:
                res[right - left] = nums[right] ** 2
                right -= 1
            else:
                res[right - left] = nums[left] ** 2
                left += 1
        res[0] = nums[left] ** 2
        return res

    # Triplet Sum to Zero
    # 15. 3Sum
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        sorted_nums = sorted(nums)

        def two_sum(nums: List[int], target: int) -> Set[Tuple[int, int]]:
            left, right = 0, len(nums) - 1
            pair_set = set()
            while left < right:
                sum = nums[left] + nums[right]
                if sum == target:
                    pair_set.add((nums[left], nums[right]))
                    left += 1
                    right -= 1
                elif sum < target:
                    left += 1
                else:
                    right -= 1
            return pair_set

        triplet_list = []
        for i, num in enumerate(sorted_nums[:-2]):
            if num > 0:
                break
            if i > 0 and num == sorted_nums[i - 1]:
                continue
            triplet_list.extend(
                [
                    [num, pair[0], pair[1]]
                    for pair in two_sum(sorted_nums[i + 1 :], -num)
                ]
            )
        return triplet_list

    # Triplet Sum Close to Target
    # 16. 3Sum Closest
    closest_num: int

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        sorted_nums = sorted(nums)
        self.closest_num = sum(sorted_nums[:3])

        def two_sum(nums: List[int], fixed_num: int):
            left, right = 0, len(nums) - 1
            while left < right:
                three_sum = fixed_num + nums[left] + nums[right]
                if three_sum == target:
                    return True
                if three_sum < target:
                    left += 1
                else:
                    right -= 1
                if abs(target - self.closest_num) > abs(target - three_sum):
                    self.closest_num = three_sum
                elif abs(target - self.closest_num) == abs(target - three_sum):
                    self.closest_num = min(self.closest_num, three_sum)
            return False

        for i, num in enumerate(sorted_nums[:-2]):
            if two_sum(sorted_nums[i + 1 :], num):
                return target

        return self.closest_num

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        n = len(nums)
        sorted_nums = sorted(nums)
        closest_num = sum(sorted_nums[:3])

        for i, num in enumerate(sorted_nums[:-2]):
            left, right = i + 1, n - 1
            while left < right:
                three_sum = num + sorted_nums[left] + sorted_nums[right]
                if three_sum == target:
                    return target

                if three_sum < target:
                    left += 1
                else:
                    right -= 1

                if abs(target - closest_num) > abs(target - three_sum):
                    closest_num = three_sum
                elif abs(target - closest_num) == abs(target - three_sum):
                    closest_num = min(closest_num, three_sum)

        return closest_num

    # Triplets with Smaller Sum
    def searchTriplets(self, arr, target):
        sorted_nums = sorted(arr)

        def count_pair(nums: List[int], max_num: int):
            count = 0
            left, right = 0, len(nums) - 1
            while left < right:
                while left < right and nums[left] + nums[right] < max_num:
                    left += 1
                count += left
                right -= 1
            return count + right * (right + 1) // 2

        return sum(
            count_pair(sorted_nums[i + 1 :], target - num)
            for i, num in enumerate(sorted_nums[:-2])
        )

    # 713. Subarray Product Less Than K
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:
            return 0

        n = len(nums)
        count = 0
        acc_product = nums[0]
        max_idx = 0
        for i, num in enumerate(nums):
            while acc_product < k:
                if max_idx == n - 1:
                    return count + (n - i) * (n - i + 1) // 2
                max_idx += 1
                acc_product *= nums[max_idx]
            count += max_idx - i
            acc_product /= num

        return count

    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:
            return 0

        count = 0
        acc_product = 1
        left = 0
        for right in range(len(nums)):
            acc_product *= nums[right]
            while acc_product >= k:
                acc_product /= nums[left]
                left += 1
            count += right - left + 1

        return count

    def findSubarrays(self, arr, target):
        if target <= 1:
            return []

        subarrays = []
        left = 0
        acc_product = 1

        for right in range(len(arr)):
            acc_product *= arr[right]
            while acc_product >= target:
                acc_product /= arr[left]
                left += 1
            for i in range(left, right + 1):
                subarrays.append(arr[i : right + 1])
        return subarrays
