from typing import Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        if l1 is None:
            return l2
        elif l2 is None:
            return l1

        first_sum = l1.val + l2.val
        if first_sum >= 10:
            first_sum -= 10
            new_next = self.addTwoNumbers(
                self.addTwoNumbers(l1.next, l2.next), ListNode(1)
            )
        else:
            new_next = self.addTwoNumbers(l1.next, l2.next)

        return ListNode(first_sum, new_next)
