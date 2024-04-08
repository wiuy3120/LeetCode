from collections import deque


class Solution:
    def checkValidString(self, s: str) -> bool:
        # Stack
        star_stack = deque()
        left_paren_stack = deque()
        for i, c in enumerate(s):
            if c == "(":
                left_paren_stack.append(i)
            elif c == "*":
                star_stack.append(i)
            elif c == ")":
                if len(left_paren_stack) > 0:
                    left_paren_stack.pop()
                elif len(star_stack) > 0:
                    star_stack.pop()
                else:
                    return False

        if len(left_paren_stack) > len(star_stack):
            return False
        else:
            while len(left_paren_stack) > 0:
                if left_paren_stack.pop() > star_stack.pop():
                    return False
            return True

    def checkValidString(self, s: str) -> bool:
        # 2 Counters to count min and max of the remaining left parentheses
        min_count = 0
        max_count = 0
        for c in s:
            if c == "(":
                min_count += 1
                max_count += 1
            elif c == "*":
                if min_count > 0:
                    min_count -= 1
                max_count += 1
            elif c == ")":
                max_count -= 1
                if max_count < 0:
                    return False
                if min_count > 0:
                    min_count -= 1

        return min_count == 0

    def checkValidString(self, s: str) -> bool:
        # Two pass
        # https://leetcode.com/problems/valid-parenthesis-string/solutions/145228/python-using-stack-20ms-beats-100-probably-the-easiest-solution/
        count = 0
        for c in s:
            if c == ")":
                if count == 0:
                    return False
                else:
                    count -= 1
            else:
                count += 1

        count = 0
        for c in s[::-1]:
            if c == "(":
                if count == 0:
                    return False
                else:
                    count -= 1
            else:
                count += 1

        return True
