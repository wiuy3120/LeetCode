# pyright: reportRedeclaration=false
import bisect
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
