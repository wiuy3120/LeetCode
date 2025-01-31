from .disjoint_set import (
    DisjointSet,
    UnknownDisjointSet,
    colorize_grid_and_get_size,
)


def adjacent_cells(i: int, j: int, m: int, n: int):
    if i > 0:
        yield i - 1, j
    if j > 0:
        yield i, j - 1
    if i < m - 1:
        yield i + 1, j
    if j < n - 1:
        yield i, j + 1
