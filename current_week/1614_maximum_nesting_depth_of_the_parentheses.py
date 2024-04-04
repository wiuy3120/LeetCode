from collections import deque


class Solution:
    def maxDepth(self, s: str) -> int:
        # left_parentheses_stack
        stack_count = 0
        max_len = 0
        for c in s:
            if c == "(":
                stack_count += 1
            if c == ")":
                if stack_count > max_len:
                    max_len = stack_count

                stack_count -= 1

        return max_len
