# pyright: reportRedeclaration=false
import bisect
import math
import random
from collections import Counter, deque
from heapq import (heapify, heappop, heappush, heappushpop, heapreplace,
                   nlargest)
from itertools import accumulate
from typing import Dict, List, Optional, Set, Tuple

from sortedcontainers import SortedList


class Solution:
    def containsDuplicate(self, nums: List[int]):
        num_set = set()
        for num in nums:
            if num in num_set:
                return True
            num_set.add(num)

        return False

    def containsDuplicate(self, nums: List[int]):
        return len(set(nums)) == len(nums)

    def checkIfPangram(self, sentence: str):
        def is_english_letter(char: str):
            if 97 <= ord(char) <= 122:
                return True
            return False

        counter = 0
        english_letter_set = set()
        for char in sentence:
            char = char.lower()
            if not is_english_letter(char):
                continue
            if char in english_letter_set:
                continue
            counter += 1
            if counter == 26:
                return True
            english_letter_set.add(char)

        return False

    def checkIfPangram(self, sentence: str):
        english_letter_set = set()
        for char in sentence:
            char = char.lower()
            if not char.isalpha():
                continue
            if char in english_letter_set:
                continue
            if len(english_letter_set) == 25:
                return True
            english_letter_set.add(char)
        return False

    def reverseVowels(self, s: str) -> str:
        vowels = "aeiouAEIOU"
        start = 0
        end = len(s) - 1
        s_list = list(s)
        while start < end:
            if s_list[start] not in vowels:
                start += 1
                continue
            if s_list[end] not in vowels:
                end -= 1
                continue
            s_list[start], s_list[end] = s_list[end], s_list[start]
            start += 1
            end -= 1
        return "".join(s_list)

    def reverseVowels(self, s: str) -> str:
        from collections import deque

        vowels = "aeiouAEIOU"
        vowel_queue = deque(c for c in s if c in vowels)
        s_list = [c if c not in vowels else vowel_queue.pop() for c in s]
        return "".join(s_list)

    def isPalindrome(self, s: str) -> bool:
        start = 0
        end = len(s) - 1
        while start < end:
            if not s[start].isalnum():
                start += 1
            elif not s[end].isalnum():
                end -= 1
            elif s[start].lower() != s[end].lower():
                return False
            else:
                start += 1
                end -= 1
        return True

    def isPalindrome(self, s: str) -> bool:
        s_list = [c.lower() for c in s if c.isalnum()]
        return s_list == s_list[::-1]

    def isPalindrome(self, s: str) -> bool:
        s = "".join(filter(str.isalnum, s.lower()))
        return all(s[i] == s[~i] for i in range(len(s) // 2))

    def isAnagram(self, s: str, t: str) -> bool:
        from collections import Counter

        if len(s) != len(t):
            return False
        counter_1 = Counter(s)
        counter_2 = Counter(t)
        return (len(counter_1 - counter_2) == 0) and (
            len(counter_2 - counter_1) == 0
        )

    def isAnagram(self, s: str, t: str) -> bool:
        from collections import Counter

        if len(s) != len(t):
            return False
        return Counter(s) == Counter(t)
