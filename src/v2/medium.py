# pyright: reportRedeclaration=false
import bisect
import math
import operator
import random
from collections import Counter, defaultdict, deque
from functools import reduce
from heapq import heapify, heappop, heappush, heappushpop, heapreplace, nlargest
from itertools import accumulate
from typing import Dict, List, Optional, Set, Tuple

from sortedcontainers import SortedList

# Binary Search


class TreeNode:
    def __init__(self, val: int = 0, left=None, right=None):
        self.val = val
        self.left: TreeNode | None = left
        self.right: TreeNode | None = right


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

    # 2981. Find Longest Special Substring That Occurs Thrice I
    def maximumLength(self, s: str) -> int:
        def has_special_substring(length: int):
            counter = {}
            for i in range(len(s) - length + 1):
                substring = s[i : i + length]

                count = counter.get(substring, 0)
                if count == 2:
                    return True
                counter[substring] = count + 1
            return False

        left, right = 1, len(s) - 2
        while left < right:
            mid = (left + right + 1) // 2
            if has_special_substring(mid):
                left = mid
            else:
                right = mid - 1

        return mid if has_special_substring(mid) else -1

    def maximumLength(self, s: str) -> int:
        counter = {s[0]: 1}
        last_special_substring_len = 1
        max_len = -1
        for i in range(1, len(s)):
            if s[i] != s[i - 1]:
                substring = s[i]
                count = counter.get(substring, 0)
                if count == 2:
                    max_len = max(max_len, 1)
                counter[substring] = count + 1
                continue

            last_special_substring_len += 1
            if last_special_substring_len <= max_len:
                continue

            for j in range(1, last_special_substring_len + 1):
                substring = s[i] * j
                count = counter.get(substring, 0)
                if count == 2:
                    max_len = max(max_len, j)
                counter[substring] = count + 1

        return max_len

    # 2779. Maximum Beauty of an Array After Applying Operation
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        min_num, max_num = min(nums), max(nums)
        counter = [0] * (max_num - min_num + 1)

        for num in nums:
            counter[max(0, num - min_num - k)] += 1
            if max_num - min_num >= num - min_num + k:
                counter[num - min_num + k] -= 1

        max_count = 0
        acc_count = 0
        for count in counter:
            acc_count += count
            max_count = max(max_count, acc_count)

        return max_count

    def maximumBeauty(self, nums: List[int], k: int) -> int:
        min_num, max_num = min(nums), max(nums)
        counter = [0] * (max_num - min_num + 1)

        for num in nums:
            counter[max(0, num - min_num - k)] += 1
            if max_num - min_num >= num - min_num + k:
                counter[num - min_num + k] -= 1

        return max(accumulate(counter))

    # 2558. Take Gifts From the Richest Pile
    def pickGifts(self, gifts: List[int], k: int) -> int:
        nums = [-num for num in gifts]
        heapify(nums)

        for _ in range(k):
            max_num = -nums[0]
            heappushpop(gifts, -math.isqrt(max_num))

        return sum(nums)

    # 2593. Find Score of an Array After Marking All Elements
    def findScore(self, nums: List[int]) -> int:
        heap = [(num, i) for i, num in enumerate(nums)]
        heapify(heap)
        marked_indices = set()
        n = len(nums)

        score = 0
        while len(marked_indices) < n:
            min_num, min_idx = heappop(heap)
            if min_idx not in marked_indices:
                score += min_num
                marked_indices.add(min_idx)
                if min_idx + 1 < n:
                    marked_indices.add(min_idx + 1)
                if min_idx - 1 >= 0:
                    marked_indices.add(min_idx - 1)

        return score

    # 769. Max Chunks To Make Sorted
    def maxChunksToSorted(self, arr: List[int]) -> int:
        res = 0
        cur_max = 0
        for i, num in enumerate(arr):
            cur_max = max(num, cur_max)
            if i == cur_max:
                res += 1
        return res

    def maxChunksToSorted(self, arr: List[int]) -> int:
        return sum(
            cur_max == i for i, cur_max in enumerate(accumulate(arr, max))
        )

    # [fav]
    # 2415. Reverse Odd Levels of Binary Tree
    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return root
        if root.left is None:
            return root

        node_list: List[TreeNode] = [root.left, root.right]

        while True:
            for i in range(len(node_list) // 2):
                node_list[i].val, node_list[-i - 1].val = (
                    node_list[-i - 1].val,
                    node_list[i].val,
                )
            if node_list[0].left is None or node_list[0].left.left is None:
                break
            new_node_list = []
            for node in node_list:
                new_node_list.extend(
                    [
                        node.left.left,
                        node.left.right,
                        node.right.left,
                        node.right.right,
                    ]
                )
            node_list = new_node_list
        return root

    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return root

        def traverse(
            left_node: TreeNode | None, right_node: TreeNode | None, level: int
        ):
            if left_node is None or right_node is None:
                return
            if (level % 2) != 0:
                left_node.val, right_node.val = right_node.val, left_node.val
            traverse(left_node.left, right_node.right, level + 1)
            traverse(left_node.right, right_node.left, level + 1)

        traverse(root.left, root.right, 1)
        return root

    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return root

        def traverse(
            left_node: TreeNode | None,
            right_node: TreeNode | None,
            is_odd: bool,
        ):
            if left_node is None or right_node is None:
                return
            if is_odd:
                left_node.val, right_node.val = right_node.val, left_node.val
            traverse(left_node.left, right_node.right, not is_odd)
            traverse(left_node.right, right_node.left, not is_odd)

        traverse(root.left, root.right, True)
        return root

    # 2471. Minimum Number of Operations to Sort a Binary Tree by Level
    def minimumOperations(self, root: Optional[TreeNode]) -> int:
        def min_swap(nums: List[int]):
            sorted_nums = sorted([(num, i) for i, num in enumerate(nums)])
            num_swaps = 0
            for i in range(len(sorted_nums)):
                _, idx = sorted_nums[i]
                while idx != i:
                    sorted_nums[i], sorted_nums[idx] = (
                        sorted_nums[idx],
                        sorted_nums[i],
                    )
                    num_swaps += 1
                    _, idx = sorted_nums[i]
            return num_swaps

        min_ops = 0
        node_list = [root]
        while len(node_list) > 0:
            min_ops += min_swap([node.val for node in node_list])
            node_list = [
                child
                for node in node_list
                for child in [node.left, node.right]
                if child is not None
            ]
        return min_ops

    # 3203. Find Minimum Diameter After Merging Two Trees
    def minimumDiameterAfterMerge(
        self, edges1: List[List[int]], edges2: List[List[int]]
    ) -> int:
        def get_diameter(edges: List[List[int]]):
            connected_dict = {i: set() for i in range(len(edges) + 1)}

            for u, v in edges:
                connected_dict[u].add(v)
                connected_dict[v].add(u)

            counter = 0
            leaf_list = [
                (edge, connected_set.pop())
                for edge, connected_set in connected_dict.items()
                if len(connected_set) == 1
            ]
            while len(leaf_list) > 0:
                counter += 1
                for leaf, parent in leaf_list:
                    connected_dict.get(parent, set()).discard(leaf)
                    connected_dict.pop(leaf)
                leaf_list = [
                    (edge, connected_set.pop())
                    for edge, connected_set in connected_dict.items()
                    if len(connected_set) == 1
                ]
            print(counter)
            return counter * 2 - (len(connected_dict) == 0)

        def farthest_node(
            start: int, connected_dict: Dict[int, List[int]]
        ) -> Tuple[int, int]:
            visited = set()
            visited.add(start)
            max_distance = 0
            farthest = start
            stack = deque([(start, 0)])
            while len(stack) > 0:
                node, distance = stack.pop()
                if distance > max_distance:
                    max_distance = distance
                    farthest = node
                for neighbor in connected_dict[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append((neighbor, distance + 1))
            return farthest, max_distance

        def get_diameter(edges: List[List[int]]):
            n = len(edges) + 1
            if n <= 2:
                return n - 1

            connected_dict = [[] for _ in range(n)]
            for u, v in edges:
                connected_dict[u].append(v)
                connected_dict[v].append(u)

            farthest, _ = farthest_node(0, connected_dict)
            _, max_distance = farthest_node(farthest, connected_dict)
            return max_distance

        def get_diameter(edges: List[List[int]]):
            n = len(edges) + 1
            if n <= 2:
                return n - 1

            graph = [[] for _ in range(n)]
            degree = [0] * n
            for v, w in edges:
                graph[v].append(w)
                graph[w].append(v)
                degree[v] += 1
                degree[w] += 1

            leaves = deque(v for v in range(n) if degree[v] == 1)
            tree_size = n
            radius = 0
            while tree_size > 2:
                for _ in range(len(leaves)):
                    leaf = leaves.popleft()
                    tree_size -= 1
                    degree[leaf] -= 1
                    for nxt in graph[leaf]:
                        degree[nxt] -= 1
                        if degree[nxt] == 1:
                            leaves.append(nxt)
                radius += 1

            return 2 * radius + (tree_size == 2)

        dia1, dia2 = get_diameter(edges1), get_diameter(edges2)

        return max(dia1, dia2, (dia1 + 1) // 2 + (dia2 + 1) // 2 + 1)

    # 515. Find Largest Value in Each Tree Row
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if root is None:
            return res

        node_list = [root]
        while len(node_list) > 0:
            res.append(max(node.val for node in node_list))
            node_list = [
                child
                for node in node_list
                for child in [node.left, node.right]
                if child is not None
            ]
        return res

    # 2559. Count Vowel Strings in Ranges
    def vowelStrings(
        self, words: List[str], queries: List[List[int]]
    ) -> List[int]:
        def is_vowel_string(word: str):
            return (word[0] in "aeiou") and (word[-1] in "aeiou")

        counter = [0] + list(
            accumulate([is_vowel_string(word) for word in words])
        )
        return [counter[end + 1] - counter[start] for start, end in queries]

    # 2270. Number of Ways to Split Array
    def waysToSplitArray(self, nums: List[int]) -> int:
        total = sum(nums)
        return sum(2 * acc >= total for acc in accumulate(nums[:-1]))

    # 1930. Unique Length-3 Palindromic Subsequences
    def countPalindromicSubsequence(self, s: str) -> int:
        char_pos_dict = {}
        for i, char in enumerate(s):
            if char not in char_pos_dict:
                char_pos_dict[char] = [i, i]
            else:
                char_pos_dict[char][1] = i

        res = 0
        for start, end in char_pos_dict.values():
            res += len(set(s[start + 1 : end]))

        return res

    def countPalindromicSubsequence(self, s: str) -> int:
        letters = "abcdefghijklmnopqrstuvwxyz"
        res = 0

        for char in letters:
            start = s.find(char)
            if start == -1:
                continue
            end = s.rfind(char)
            if start >= end:
                continue

            v = [False] * 128
            distinct_count = 0
            for i in range(start + 1, end):
                if not v[ord(s[i])]:
                    v[ord(s[i])] = True
                    distinct_count += 1
                    if distinct_count == 26:
                        break
            res += distinct_count
        return res

    def countPalindromicSubsequence(self, s: str) -> int:
        char_pos_dict = {}
        for i, char in enumerate(s):
            if char not in char_pos_dict:
                char_pos_dict[char] = [i, i]
            else:
                char_pos_dict[char][1] = i

        def distinct_char(start: int, end: int, s: str) -> int:
            visited = [False] * 26
            count = 0
            for char in s[start + 1 : end]:
                if visited[ord(char) - ord("a")]:
                    continue
                visited[ord(char) - ord("a")] = True
                count += 1
                if count == 26:
                    return 26
            return count

        res = 0
        for start, end in char_pos_dict.values():
            res += distinct_char(start, end, s)
        return res

    def countPalindromicSubsequence(self, s: str) -> int:
        char_range = [None] * 26
        for i, char in enumerate(s):
            idx = ord(char) - ord("a")
            if char_range[idx] is None:
                char_range[idx] = [i, i]
            else:
                char_range[idx][1] = i

        def distinct_char(start: int, end: int, s: str) -> int:
            visited = [False] * 26
            count = 0
            for char in s[start + 1 : end]:
                if visited[ord(char) - ord("a")]:
                    continue
                visited[ord(char) - ord("a")] = True
                count += 1
                if count == 26:
                    return 26
            return count

        return sum(
            distinct_char(range[0], range[1], s)
            for range in char_range
            if range is not None
        )

    # 2381. Shifting Letters II
    def shiftingLetters(self, s: str, shifts: List[List[int]]) -> str:
        def shift_char(char: str, shift: int) -> str:
            return chr((ord(char) - ord("a") + shift) % 26 + ord("a"))

        shift_count = [0] * (len(s) + 1)
        for start, end, direction in shifts:
            shift = 1 if direction == 1 else -1
            shift_count[start] += shift
            shift_count[end + 1] -= shift

        return "".join(
            shift_char(char, shift)
            for char, shift in zip(s, accumulate(shift_count[:-1]))
        )

    # 2185. Counting Words With a Given Prefix
    def prefixCount(self, words: List[str], pref: str) -> int:
        return sum(1 for word in words if word.startswith(pref))

    # 2116. Check if a Parentheses String Can Be Valid
    def canBeValid(self, s: str, locked: str) -> bool:
        if len(s) % 2 != 0:
            return False

        if locked[-1] == "1" and s[-1] == "(":
            return False

        locked_stack = deque()
        unlocked_stack = deque()
        for i, char in enumerate(s):
            if locked[i] == "0":
                unlocked_stack.append(i)
                continue
            if char == "(":
                locked_stack.append(i)
                continue
            if len(locked_stack) > 0:
                locked_stack.pop()
                continue
            if len(unlocked_stack) == 0:
                return False
            unlocked_stack.pop()

        # while len(locked_stack) > 0:
        #     if len(unlocked_stack) == 0:
        #         return False
        #     if locked_stack.pop() > unlocked_stack.pop():
        #         return False
        # return True

        while (
            len(locked_stack) > 0
            and len(unlocked_stack) > 0
            and locked_stack[-1] < unlocked_stack[-1]
        ):
            locked_stack.pop()
            unlocked_stack.pop()
        return len(locked_stack) == 0

    # 3223. Minimum Length of String After Operations
    def minimumLength(self, s: str) -> int:
        return sum(1 + (count + 1) % 2 for count in Counter(s).values())

    def minimumLength(self, s: str) -> int:
        return sum(1 if count & 1 else 2 for count in Counter(s).values())

    def minimumLength(self, s: str) -> int:
        char_set = {
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
        }
        res = 0
        for char in char_set:
            count = s.count(char)
            if count > 0:
                res += 1 if count & 1 else 2
        return res

    # 2657. Find the Prefix Common Array of Two Arrays
    def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
        appeared = [False] * len(A)
        common_counter = [0] * len(A)
        for i, (num1, num2) in enumerate(zip(A, B)):
            if appeared[num1 - 1]:
                common_counter[i] += 1
            else:
                appeared[num1 - 1] = True

            if appeared[num2 - 1]:
                common_counter[i] += 1
            else:
                appeared[num2 - 1] = True
        return list(accumulate(common_counter))

    def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
        num_set = set()
        common_counter = [0] * len(A)
        for i, (num1, num2) in enumerate(zip(A, B)):
            num_set.add(num1)
            num_set.add(num2)
            common_counter[i] = 2 * (i + 1) - len(num_set)
        return common_counter

    # 2429. Minimize XOR
    def minimizeXor(self, num1: int, num2: int) -> int:
        ones = sum(int(bit) for bit in bin(num2)[2:])
        if ones >= len(bin(num1)[2:]):
            return (2 << (ones - 1)) - 1
        res_bin = ""
        num1_bin = bin(num1)[2:]
        for i, bit in enumerate(num1_bin):
            if ones >= (len(num1_bin) - i):
                break
            if ones == 0:
                res_bin += "0"
                continue
            if bit == "1":
                ones -= 1
            res_bin += bit
        res_bin += "1" * ones
        return int(res_bin, 2)

    # 2425. Bitwise XOR of All Pairings
    def xorAllNums(self, nums1: List[int], nums2: List[int]) -> int:
        res = reduce(operator.xor, nums1) if len(nums2) % 2 == 1 else 0
        res ^= reduce(operator.xor, nums2) if len(nums1) % 2 == 1 else 0
        return res

    def xorAllNums(self, nums1: List[int], nums2: List[int]) -> int:
        return (reduce(operator.xor, nums1) if len(nums2) % 2 == 1 else 0) ^ (
            reduce(operator.xor, nums2) if len(nums1) % 2 == 1 else 0
        )


if __name__ == "__main__":
    solution = Solution()
    edges1 = [[0, 1], [0, 2], [0, 3]]
    edges2 = [[0, 1]]
    print(solution.minimumDiameterAfterMerge(edges1, edges2))
    edges1 = [[3, 0], [2, 1], [2, 3]]
    edges2 = [
        [0, 1],
        [0, 4],
        [3, 5],
        [6, 3],
        [7, 6],
        [2, 7],
        [0, 2],
        [8, 0],
        [8, 9],
    ]
    print(solution.minimumDiameterAfterMerge(edges1, edges2))
    edges1 = [
        [0, 1],
        [2, 0],
        [3, 2],
        [3, 6],
        [8, 7],
        [4, 8],
        [5, 4],
        [3, 5],
        [3, 9],
    ]
    edges2 = [[0, 1], [0, 2], [0, 3]]
    print(solution.minimumDiameterAfterMerge(edges1, edges2))
