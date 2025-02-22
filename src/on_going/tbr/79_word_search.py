from typing import List


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        for i, row in enumerate(board):
            for j, c in enumerate(row):
                return False

        return True


board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
word = "ABCCED"
print(Solution().exist(board, word))

word = "SEE"
print(Solution().exist(board, word))

word = "ABCB"
print(Solution().exist(board, word))
