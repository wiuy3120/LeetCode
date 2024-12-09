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
    # 2290. Minimum Obstacle Removal to Reach Corner
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        visited_cells = [[False] * n for _ in range(m)]

        def new_neighbors(row: int, col: int):
            neighbors = [
                (row + i, col + j)
                for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            ]
            return [
                (row, col)
                for row, col in neighbors
                if (0 <= row < m)
                and (0 <= col < n)
                and (not visited_cells[row][col])
            ]

        priority_queue = [(grid[0][0], 0, 0)]
        while True:
            weight, row, col = heappop(priority_queue)
            if visited_cells[row][col]:
                continue
            visited_cells[row][col] = True

            new_cells = new_neighbors(row, col)

            for row, col in new_cells:
                if row == m - 1 and col == n - 1:
                    return weight + grid[row][col]
                heappush(priority_queue, (weight + grid[row][col], row, col))

    # 2577. Minimum Time to Visit a Cell In a Grid
    def minimumTime(self, grid: List[List[int]]) -> int:
        if grid[0][1] > 1 and grid[1][0] > 1:
            return -1

        m = len(grid)
        n = len(grid[0])
        visited_cells = [[False] * n for _ in range(m)]

        def new_neighbors(row: int, col: int):
            STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            neighbors = [(row + i, col + j) for i, j in STEPS]
            return [
                (row, col)
                for row, col in neighbors
                if (0 <= row < m)
                and (0 <= col < n)
                and (not visited_cells[row][col])
            ]

        def min_time(row: int, col: int):
            pos_min = row + col
            value = grid[row][col]
            return max(pos_min, value + ((pos_min ^ value) & 1))

        priority_queue = [(0, 0, 0)]
        while True:
            weight, row, col = heappop(priority_queue)
            new_cells = new_neighbors(row, col)

            for row, col in new_cells:
                new_weight = max(min_time(row, col), weight + 1)
                if (row == m - 1) and (col == n - 1):
                    return new_weight
                heappush(priority_queue, (new_weight, row, col))
                visited_cells[row][col] = True

    # 2097. Valid Arrangement of Pairs
    def validArrangement(self, pairs: List[List[int]]) -> List[List[int]]:
        graph = defaultdict(list)
        degree = defaultdict(int)  # net out degree
        for x, y in pairs:
            graph[x].append(y)
            degree[x] += 1
            degree[y] -= 1

        for k in degree:
            if degree[k] == 1:
                x = k
                break

        ans = []

        def fn(x):
            """Return Eulerian path via dfs."""
            while graph[x]:
                fn(graph[x].pop())
            ans.append(x)

        fn(x)
        ans.reverse()
        return [[ans[i], ans[i + 1]] for i in range(len(ans) - 1)]
