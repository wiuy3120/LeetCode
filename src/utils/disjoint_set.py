from typing import Callable, List


class DisjointSet:
    def __init__(self, N):
        # Initialize DSU class, size of each component will be one and each node
        # will be representative of its own.
        self.N = N
        self.size = [1] * N
        self.representative = list(range(N))

    def find(self, node):
        # Returns the ultimate representative of the node.
        if self.representative[node] == node:
            return node
        self.representative[node] = self.find(self.representative[node])
        return self.representative[node]

    def union(self, nodeOne, nodeTwo):
        # Returns true if node nodeOne and nodeTwo belong to different component
        # and update the representatives accordingly, otherwise returns false.
        nodeOne = self.find(nodeOne)
        nodeTwo = self.find(nodeTwo)

        if nodeOne == nodeTwo:
            return False

        if self.size[nodeOne] > self.size[nodeTwo]:
            self.representative[nodeTwo] = nodeOne
            self.size[nodeOne] += self.size[nodeTwo]
        else:
            self.representative[nodeOne] = nodeTwo
            self.size[nodeTwo] += self.size[nodeOne]
        return True


class UnknownDisjointSet(DisjointSet):
    """
    A disjoint set data structure implementation
    that does not require knowing the number of sets in advance.

    This class extends the DisjointSet class
    and provides functionality to dynamically manage disjoint sets
    without predefining the number of elements or sets.

    Attributes:
        size (dict): A dictionary that keeps track of the size of each set.
        representative (dict): A dictionary that maps each element
            to its representative (or root) in the set.

    Methods:
        __init__(): Initializes the disjoint set data structure.
    """

    def __init__(self):
        self.size = {}
        self.representative = {}


class DisjointSetWithSize:

    def __init__(self):
        self.size = {}
        self.representative = {}

    def find(self, node):
        # Returns the ultimate representative of the node.
        if self.representative[node] == node:
            return node
        self.representative[node] = self.find(self.representative[node])
        return self.representative[node]

    def add(self, node):
        # Add a new node to the disjoint set data structure.
        self.representative[node] = node
        self.size[node] = 1

    def increase_size(self, node):
        # Increase the size of the set that contains the given node.
        self.size[self.find(node)] += 1

    def union(self, nodeOne, nodeTwo, increase_size: bool = True):
        # Returns true if node nodeOne and nodeTwo belong to different component
        # and update the representatives accordingly, otherwise returns false.
        nodeOne = self.find(nodeOne)
        nodeTwo = self.find(nodeTwo)

        if nodeOne == nodeTwo:
            if increase_size:
                self.increase_size(nodeOne)
            return nodeOne

        if self.size[nodeOne] > self.size[nodeTwo]:
            parent = nodeOne
            self.representative[nodeTwo] = nodeOne
            # self.size[nodeOne] += self.size[nodeTwo]

        else:
            parent = nodeTwo
            self.representative[nodeOne] = nodeTwo
            # self.size[nodeTwo] += self.size[nodeOne]

        if increase_size:
            self.size[parent] = self.size[nodeOne] + self.size[nodeTwo] + 1
        return parent


def _backward_adjacent_cells(i: int, j: int):
    if i > 0:
        yield i - 1, j
    if j > 0:
        yield i, j - 1


def colorize_grid_and_get_size(
    grid: List[List[int]], is_color: Callable[[int, int], bool]
):
    m, n = len(grid), len(grid[0])
    new_grid = [[0] * n for _ in range(m)]
    groups = DisjointSetWithSize()

    num_groups = 0
    for row in range(m):
        for col in range(n):
            if not is_color(row, col):
                new_grid[row][col] = 0
                continue
            color_adjacent_cells = [
                cell
                for cell in _backward_adjacent_cells(row, col)
                if is_color(*cell)
            ]
            match color_adjacent_cells:
                case []:
                    num_groups += 1
                    groups.add(num_groups)
                    new_grid[row][col] = num_groups
                case [(i, j)]:
                    groups.increase_size(new_grid[i][j])
                    new_grid[row][col] = groups.find(new_grid[i][j])
                case [(i, j), (k, l)]:
                    parent = groups.union(
                        new_grid[i][j], new_grid[k][l], increase_size=True
                    )
                    new_grid[row][col] = parent
    return new_grid, groups
