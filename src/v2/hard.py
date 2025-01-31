# pyright: reportRedeclaration=false
import bisect
import math
import random
from collections import Counter, defaultdict, deque
from heapq import heapify, heappop, heappush, heappushpop, heapreplace, nlargest
from itertools import accumulate
from typing import Dict, List, Optional, Set, Tuple

from sortedcontainers import SortedList

from utils import DisjointSet


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

    # [fav]
    # 768. Max Chunks To Make Sorted II
    def maxChunksToSorted(self, arr: List[int]) -> int:
        # max_arr = list(accumulate(arr[:-1], max))
        min_arr = list(accumulate(reversed(arr[1:]), min))
        res = 1
        for i, cur_max in enumerate(accumulate(arr[:-1], max)):
            next_min = min_arr[-1 - i]
            if cur_max <= next_min:
                res += 1
        return res

    def maxChunksToSorted(self, arr: List[int]) -> int:
        min_arr = list(accumulate(reversed(arr[1:]), min))
        return (
            sum(
                cur_max <= min_arr[-1 - i]
                for i, cur_max in enumerate(accumulate(arr[:-1], max))
            )
            + 1
        )

    # [fav]
    # 2872. Maximum Number of K-Divisible Components
    def maxKDivisibleComponents(
        self, n: int, edges: List[List[int]], values: List[int], k: int
    ) -> int:
        adjacent_dict = {i: set() for i in range(n)}
        for edge1, edge2 in edges:
            adjacent_dict[edge1].add(edge2)
            adjacent_dict[edge2].add(edge1)

        stack = deque()
        for node, adj_set in adjacent_dict.items():
            if len(adj_set) == 1:
                stack.append(node)

        num_comps = 1
        for _ in range(n - 1):
            node = stack.pop()
            adj_set = adjacent_dict[node]
            node_value = values[node]
            if node_value % k == 0:
                num_comps += 1
            adj_node = adj_set.pop()
            values[adj_node] += node_value
            adjacent_dict[adj_node].remove(node)
            if len(adjacent_dict[adj_node]) == 1:
                stack.append(adj_node)

        return num_comps

    # 2493. Divide Nodes Into the Maximum Number of Groups
    def magnificentSets(self, n: int, edges: List[List[int]]) -> int:
        # Function to calculate the number of groups
        # for a given component starting from srcNode
        def get_number_of_groups(
            adjacencies_list: List[List[int]], node: int, n: int
        ):
            nodes_queue = deque()
            layer_seen = [-1] * n
            nodes_queue.append(node)
            layer_seen[node] = 0
            deepest_layer = 0

            # Perform BFS to calculate the number of layers (groups)
            while nodes_queue:
                num_of_nodes_in_layer = len(nodes_queue)
                for _ in range(num_of_nodes_in_layer):
                    current_node = nodes_queue.popleft()
                    for neighbor in adjacencies_list[current_node]:
                        # If neighbor hasn't been visited,
                        # assign it to the next layer
                        if layer_seen[neighbor] == -1:
                            layer_seen[neighbor] = deepest_layer + 1
                            nodes_queue.append(neighbor)
                        else:
                            # If the neighbor is already in the same layer,
                            # return -1 (invalid partition)
                            if layer_seen[neighbor] == deepest_layer:
                                return -1
                deepest_layer += 1
            return deepest_layer

        adjacencies_list = [[] for _ in range(n)]
        components = DisjointSet(n)

        # Build the adjacency list and apply Union-Find for each edge
        for edge in edges:
            adjacencies_list[edge[0] - 1].append(edge[1] - 1)
            adjacencies_list[edge[1] - 1].append(edge[0] - 1)
            components.union(edge[0] - 1, edge[1] - 1)

        num_of_groups_for_component = {}

        # For each node, calculate the maximum number of groups
        # for its component
        for node in range(n):
            number_of_groups = get_number_of_groups(adjacencies_list, node, n)
            if number_of_groups == -1:
                return -1  # If invalid split, return -1
            root_node = components.find(node)
            num_of_groups_for_component[root_node] = max(
                num_of_groups_for_component.get(root_node, 0), number_of_groups
            )

        # Calculate the total number of groups across all components
        total_number_of_groups = sum(num_of_groups_for_component.values())
        return total_number_of_groups

    # Main function to calculate the maximum number of magnificent sets
    def magnificentSets(self, n, edges):
        # Create adjacency list for the graph
        adj_list = [[] for _ in range(n)]
        for edge in edges:
            # Transition to 0-index
            adj_list[edge[0] - 1].append(edge[1] - 1)
            adj_list[edge[1] - 1].append(edge[0] - 1)

        # Initialize color array to -1
        colors = [-1] * n

        # Check if the graph is bipartite
        for node in range(n):
            if colors[node] != -1:
                continue
            # Start coloring from uncolored nodes
            colors[node] = 0
            if not self._is_bipartite(adj_list, node, colors):
                return -1

        # Calculate the longest shortest path for each node
        distances = [
            self._get_longest_shortest_path(adj_list, node, n)
            for node in range(n)
        ]

        # Calculate the total maximum number of groups across all components
        max_number_of_groups = 0
        visited = [False] * n
        for node in range(n):
            if visited[node]:
                continue
            # Add the number of groups for this component to the total
            max_number_of_groups += self._get_number_of_groups_for_component(
                adj_list, node, distances, visited
            )

        return max_number_of_groups

    # Checks if the graph is bipartite starting from the given node
    def _is_bipartite(self, adj_list, node, colors):
        for neighbor in adj_list[node]:
            # If a neighbor has the same color as the current node,
            # the graph is not bipartite
            if colors[neighbor] == colors[node]:
                return False
            # If the neighbor is already colored, skip it
            if colors[neighbor] != -1:
                continue
            # Assign the opposite color to the neighbor
            colors[neighbor] = (colors[node] + 1) % 2
            # Recursively check bipartiteness for the neighbor;
            # return false if it fails
            if not self._is_bipartite(adj_list, neighbor, colors):
                return False
        # If all neighbors are properly colored, return true
        return True

    # Computes the longest shortest path (height) in the graph
    # starting from the source node
    def _get_longest_shortest_path(self, adj_list, src_node, n):
        # Initialize a queue for BFS and a visited array
        nodes_queue = deque([src_node])
        visited = [False] * n
        visited[src_node] = True
        distance = 0

        # Perform BFS layer by layer
        while nodes_queue:
            # Process all nodes in the current layer
            for _ in range(len(nodes_queue)):
                current_node = nodes_queue.popleft()
                # Visit all unvisited neighbors of the current node
                for neighbor in adj_list[current_node]:
                    if visited[neighbor]:
                        continue
                    visited[neighbor] = True
                    nodes_queue.append(neighbor)
            # Increment the distance for each layer
            distance += 1

        # Return the total distance (longest shortest path)
        return distance

    # Calculates the maximum number of groups for a connected component
    def _get_number_of_groups_for_component(
        self, adj_list, node, distances, visited
    ):
        # Start with the distance of the current node as the maximum
        max_number_of_groups = distances[node]
        visited[node] = True

        # Recursively calculate the maximum for all unvisited neighbors
        for neighbor in adj_list[node]:
            if visited[neighbor]:
                continue
            max_number_of_groups = max(
                max_number_of_groups,
                self._get_number_of_groups_for_component(
                    adj_list, neighbor, distances, visited
                ),
            )
        return max_number_of_groups
