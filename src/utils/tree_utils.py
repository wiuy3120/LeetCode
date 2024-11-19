# support for maintaining a list in sorted order
# without having to sort the list after each insertion
import bisect
import random
from collections import Counter, deque
from heapq import heapify, heappop, heappush, heapreplace, nlargest
from typing import Dict, List, Optional, Set, Tuple

from sortedcontainers import SortedList


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def list2tree(num_list: List[Optional[int]]):
    if num_list[0] is None:
        return None

    root = TreeNode(num_list[0])
    upper_nodes = [root]
    i = 1
    n = len(num_list)
    while i < n:
        new_upper_nodes = []
        max_idx = min(n, i + len(upper_nodes) * 2)
        for j, num in enumerate(num_list[i:max_idx]):
            parent_node = upper_nodes[j // 2]
            new_node = TreeNode(num) if num is not None else None
            new_upper_nodes.append(new_node)
            if j % 2 == 0:
                parent_node.left = new_node
            else:
                parent_node.right = new_node
        upper_nodes = new_upper_nodes
        i = max_idx

    return root
