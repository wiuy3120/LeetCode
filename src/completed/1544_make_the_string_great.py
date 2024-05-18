from collections import deque


class Solution:
    def makeGood(self, s: str) -> str:
        great_string = ""
        for c in s:
            if len(great_string) == 0:
                great_string = c
            elif (
                great_string[-1].lower() == c.lower() and great_string[-1] != c
            ):
                great_string = great_string[:-1]
            else:
                great_string += c

        return great_string

    def makeGoodStack(self, s: str) -> str:
        # Best
        great_stack = deque()
        for c in s:
            if len(great_stack) == 0:
                great_stack.append(c)
            elif great_stack[-1].lower() == c.lower() and great_stack[-1] != c:
                great_stack.pop()
            else:
                great_stack.append(c)

        return "".join(great_stack)
