# https://leetcode.com/problems/two-sum/

from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        result = [i for i, num in enumerate(nums) if num * 2 == target]
        if len(result) == 2:
            return result

        result = []
        target_nums = set(target - num for num in nums if num * 2 != target)

        for i, value in enumerate(nums):
            if value in target_nums:
                result.append(i)

        return result
