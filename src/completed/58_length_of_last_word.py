class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        tail = len(s) - 1
        result = 0
        while s[tail] == " ":
            tail -= 1

        while s[tail] != " ":
            result += 1
            if tail == 0:
                return result
            tail -= 1

        return result
