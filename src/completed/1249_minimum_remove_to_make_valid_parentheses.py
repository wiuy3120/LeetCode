from collections import deque

from sortedcontainers import SortedList


class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        stack = deque()
        remove_indexs = SortedList()
        for i, c in enumerate(s):
            if c == ")":
                if len(stack) == 0:
                    remove_indexs.add(i)
                else:
                    stack.pop()
            elif c == "(":
                stack.append(i)

        remove_indexs.update(stack)

        if len(remove_indexs) == 0:
            return s

        output = s[: remove_indexs[0]]
        for i in range(len(remove_indexs) - 1):
            output += s[(remove_indexs[i] + 1) : remove_indexs[i + 1]]
        output += s[(remove_indexs[-1] + 1) :]

        return output

    def minRemoveToMakeValid(self, s: str) -> str:
        """
        Construct new string
        and a stack containing remaining right panrenthese indices
        to remove later
        """
        stack = deque()
        output = ""
        removed_count = 0
        for i, c in enumerate(s):
            if c == ")":
                if len(stack) == 0:
                    removed_count += 1
                else:
                    stack.pop()
                    output += c
            elif c == "(":
                stack.append(i - removed_count)
                output += c
            else:
                output += c

        if len(stack) == 0:
            return output

        remove_indexs = list(stack)

        final_output = output[: remove_indexs[0]]
        for i in range(len(remove_indexs) - 1):
            final_output += output[remove_indexs[i] + 1 : remove_indexs[i + 1]]
        final_output += output[remove_indexs[-1] + 1 :]

        return final_output

    def minRemoveToMakeValid(self, s: str) -> str:
        # convert string to list
        s = list(s)
        stack = []
        for i, char in enumerate(s):
            if char == "(":
                stack.append(i)
            elif char == ")":
                if stack:
                    stack.pop()
                else:
                    s[i] = ""
        while stack:
            s[stack.pop()] = ""
        return "".join(s)


s = "lee(t(c)o)de)"
print(Solution().minRemoveToMakeValid(s))

s = "())()((("
print(Solution().minRemoveToMakeValid(s))

s = "))(("
print(Solution().minRemoveToMakeValid(s))
