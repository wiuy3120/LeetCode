# pyright: reportRedeclaration=false
import bisect
import math
import random
from collections import Counter, deque, defaultdict
from heapq import (
    heapify,
    heappop,
    heappush,
    heappushpop,
    heapreplace,
    nlargest,
)
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
