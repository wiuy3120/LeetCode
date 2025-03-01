# pyright: reportRedeclaration=false
import bisect
import enum
import math
import random
from collections import Counter, defaultdict, deque
from heapq import heapify, heappop, heappush, heappushpop, heapreplace, nlargest
from itertools import accumulate
from typing import Dict, List, Optional, Set, Tuple

from sortedcontainers import SortedList


class Solution:
    # 1346. Check If N and Its Double Exist
    def checkIfExist(self, arr: List[int]) -> bool:
        num_set = set()

        for num in arr:
            if num * 2 in num_set:
                return True
            if num % 2 == 0 and num // 2 in num_set:
                return True
            num_set.add(num)

        return False

    # 1455. Check If a Word Occurs As a Prefix of Any Word in a Sentence
    def isPrefixOfWord(self, sentence: str, searchWord: str) -> int:
        word_list = sentence.split(" ")
        for i, word in enumerate(word_list):
            if word.startswith(searchWord):
                return i + 1

        return -1

    # 1422. Maximum Score After Splitting a String
    def maxScore(self, s: str) -> int:
        n = len(s)
        zero_counter = 0
        max_score = 0
        total_zeros = sum([1 for char in s if char == "0"])
        for i, char in enumerate(s[:-1]):
            if char == "0":
                zero_counter += 1
            max_score = max(
                max_score, n - i - 1 - total_zeros + 2 * zero_counter
            )
        return max_score

    def maxScore(self, s: str) -> int:
        zero_counter = 0
        max_score = 0
        total_ones = sum([1 for char in s if char == "1"])
        for i, char in enumerate(s[:-1]):
            if char == "0":
                zero_counter += 1
            max_score = max(max_score, total_ones - i - 1 + 2 * zero_counter)
        return max_score

    def maxScore(self, s: str) -> int:
        zero_counter, one_counter = 0
        max_score = -1
        for i, char in enumerate(s[:-1]):
            if char == "0":
                zero_counter += 1
            else:
                one_counter += 1
            max_score = max(max_score, 2 * zero_counter - i - 1)
        one_counter += 1 if s[-1] == "1" else 0
        return max_score + one_counter

    # 1408. String Matching in an Array
    def stringMatching(self, words: List[str]) -> List[str]:
        sentence = " ".join(words)
        return [word for word in words if sentence.count(word) > 1]

    # 3042. Count Prefix and Suffix Pairs I
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        def is_presuffix(str1: str, str2: str):
            return str2.startswith(str1) and str2.endswith(str1)

        return sum(
            is_presuffix(str1, str2)
            for i, str1 in enumerate(words[:-1])
            for str2 in words[i + 1 :]
        )

    # 1769. Minimum Number of Operations to Move All Balls to Each Box
    def minOperations(self, boxes: str) -> List[int]:
        moves = [0 if char == "0" else 1 for char in boxes]
        left_moves = accumulate([0] + list(accumulate(moves[:-1])))
        right_moves = list(accumulate([0] + list(accumulate(moves[:0:-1]))))[
            ::-1
        ]
        return [left + right for left, right in zip(left_moves, right_moves)]

    # 3151. Special Array I
    def isArraySpecial(self, nums: List[int]) -> bool:
        for i in range(1, len(nums)):
            if (nums[i] & 1) == (nums[i - 1] & 1):
                return False
        return True

    def isArraySpecial(self, nums: List[int]) -> bool:
        return not any(
            (nums[i] & 1) == (nums[i - 1] & 1) for i in range(1, len(nums))
        )

    def isArraySpecial(self, nums: List[int]) -> bool:
        return all(
            (nums[i] & 1) != (nums[i - 1] & 1) for i in range(1, len(nums))
        )

    def isArraySpecial(self, nums: List[int]) -> bool:
        return all(
            False
            for i in range(1, len(nums))
            if (nums[i] & 1) == (nums[i - 1] & 1)
        )

    # 1752. Check if Array Is Sorted and Rotated
    def check(self, nums: List[int]) -> bool:
        is_rotated = False
        for i in range(1, len(nums)):
            if nums[i] < nums[i - 1]:
                if is_rotated:
                    return False
                is_rotated = True
        return not is_rotated or nums[-1] <= nums[0]

    # 3105. Longest Strictly Increasing or Strictly Decreasing Subarray
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        res = 1
        direction = 0
        left_pointer = 0
        for right_pointer in range(1, len(nums)):
            if nums[right_pointer] == nums[right_pointer - 1]:
                left_pointer = right_pointer
                direction = 0
            elif nums[right_pointer] > nums[right_pointer - 1]:
                if direction == -1:
                    left_pointer = right_pointer - 1
                direction = 1
            else:
                if direction == 1:
                    left_pointer = right_pointer - 1
                direction = -1
            res = max(res, right_pointer - left_pointer + 1)
        return res

    # 1790. Check if One String Swap Can Make Strings Equal
    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        if len(s1) != len(s2):
            return False
        diff_count = 0
        diff_chars = []
        for c1, c2 in zip(s1, s2):
            if c1 != c2:
                diff_count += 1
                if diff_count > 2:
                    return False
                diff_chars.append((c1, c2))
        if diff_count == 0:
            return True
        if diff_count == 1:
            return False
        return diff_chars[0] == diff_chars[1][::-1]

    # 1800. Maximum Ascending Subarray Sum
    def maxAscendingSum(self, nums: List[int]) -> int:
        max_sum = current_sum = nums[0]
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                current_sum += nums[i]
            else:
                current_sum = nums[i]
            max_sum = max(max_sum, current_sum)
        return max_sum

    def maxAscendingSum(self, nums: List[int]) -> int:
        max_sum = current_sum = nums[0]
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                current_sum += nums[i]
            else:
                max_sum = max(max_sum, current_sum)
                current_sum = nums[i]
        return max(max_sum, current_sum)

    # 3174. Clear Digits
    def clearDigits(self, s: str) -> str:
        res = ""
        num_digits = 0
        for char in s[::-1]:
            if char.isdigit():
                num_digits += 1
            else:
                if num_digits != 0:
                    num_digits -= 1
                else:
                    res = char + res
        return res

    # 2460. Apply Operations to an Array
    def applyOperations(self, nums: List[int]) -> List[int]:
        left = 0
        for right in range(1, len(nums)):
            if nums[left] == 0:
                nums[left] = nums[right]
                nums[right] = 0
                continue
            if nums[left] == nums[right]:
                nums[left] *= 2
                nums[right] = 0
            elif left + 1 != right:
                nums[left + 1] = nums[right]
                nums[right] = 0
            left += 1
        return nums

    def applyOperations(self, nums: List[int]) -> List[int]:
        left = 0
        for right in range(len(nums) - 1):
            if nums[right] == nums[right + 1]:
                nums[right] *= 2
                nums[right + 1] = 0
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
        if nums[-1] != 0:
            nums[left], nums[-1] = nums[-1], nums[left]
        return nums
