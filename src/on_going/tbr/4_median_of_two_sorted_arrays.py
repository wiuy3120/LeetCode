import bisect
from typing import List


class Solution:
    def findMedianNonEmptySortedArray(self, nums: List[int]) -> float:
        n = len(nums)
        if n % 2 == 0:
            return (nums[n // 2] + nums[n // 2 - 1]) / 2
        else:
            return nums[n // 2]

    def findMedianSortedArraysConcatenatedCase(
        self, nums1: List[int], nums2: List[int]
    ) -> float:
        n = len(nums1)
        total_len = n + len(nums2)

        def get_value(i: int):
            if i >= n:
                return nums2[i - n]
            else:
                return nums1[i]

        if total_len % 2 == 0:
            return (
                get_value(total_len // 2) + get_value(total_len // 2 - 1)
            ) / 2
        else:
            return get_value(total_len // 2)

    def rightClosestValueIndex(self, nums: List[int], value: int):
        return bisect.bisect_right(nums, value)

    def leftClosestValueIndex(self, nums: List[int], value: int):
        return bisect.bisect_left(nums, value)

    # def findMedianSortedArraysDominatedCase(
    #     self, nums1: List[int], nums2: List[int], direction: str
    # ) -> float:

    def findMedianSortedArrays(
        self, nums1: List[int], nums2: List[int]
    ) -> float:
        if len(nums1) < len(nums2):
            return self.findMedianSortedArrays(nums2, nums1)

        # n = len(nums1)
        # if n == 0:
        #     return self.findMedianNonEmptySortedArray(nums2)
        m = len(nums2)
        if m == 0:
            return self.findMedianNonEmptySortedArray(nums1)
        n = len(nums1)

        if nums1[-1] <= nums2[0]:
            return self.findMedianSortedArraysConcatenatedCase(arr_1, arr_2)
        if nums2[-1] <= nums1[0]:
            return self.findMedianSortedArraysConcatenatedCase(arr_2, arr_1)

        total_len = n + m
        half_len = total_len // 2
        if total_len % 2 == 1:
            if nums2[-1] < nums1[-1]:
                count = self.leftClosestValueIndex(nums1, nums2[-1]) + m - 1
                if count == half_len:
                    return nums2[-1]
                elif count < half_len:
                    return nums1[-half_len - 1]

            else:
                count = self.rightClosestValueIndex(nums1, nums2[0]) + m - 1
                if count == half_len:
                    return nums2[0]
                elif count < half_len:
                    return nums1[half_len]

        else:
            if nums2[-1] < nums1[-1]:
                count = n - self.leftClosestValueIndex(nums1, nums2[-1]) + m
                if count < half_len:
                    return (nums1[-half_len] + nums1[-half_len - 1]) / 2
                elif count == half_len:
                    left = nums2[-1]
                    right = nums1[self.leftClosestValueIndex(nums1, nums2[-1])]
                    return (left + right) / 2
                elif count == half_len + 1:
                    left = max(
                        nums2[-2],
                        nums1[self.leftClosestValueIndex(nums1, nums2[-1]) - 1],
                    )
                    right = nums2[-1]
                    return (left + right) / 2
                elif count < half_len:
                    return (nums1[-half_len - 1] + nums1[-half_len]) / 2

            else:
                count = n - self.rightClosestValueIndex(nums1, nums2[0]) + m
                if count < half_len:
                    return (nums1[half_len] + nums1[half_len - 1]) / 2
                elif count == half_len:
                    left = nums1[
                        self.leftClosestValueIndex(nums1, nums2[0]) - 1
                    ]
                    right = nums2[0]
                    return (left + right) / 2
                elif count == half_len + 1:
                    left = nums2[0]
                    right = max(
                        nums2[1],
                        nums1[self.leftClosestValueIndex(nums1, nums2[0])],
                    )
                    return (left + right) / 2
                elif count < half_len:
                    return (nums1[half_len - 1] + nums1[half_len]) / 2


arr_1 = []
arr_2 = [3, 4, 5, 6]
print(Solution().findMedianSortedArrays(arr_1, arr_2))

arr_1 = [2, 3, 4, 5, 6]
arr_2 = []
print(Solution().findMedianSortedArrays(arr_1, arr_2))

arr_1 = [1, 3]
arr_2 = [2]
print(Solution().findMedianSortedArrays(arr_1, arr_2))

arr_1 = [1, 2]
arr_2 = [3, 4, 5, 6]
print(Solution().findMedianSortedArrays(arr_1, arr_2))

arr_1 = [1, 2, 3]
arr_2 = [3, 3, 4, 5, 6]
print(Solution().findMedianSortedArrays(arr_1, arr_2))


arr_1 = [1, 2, 5, 9, 11]
arr_2 = [10, 12]
print(Solution().findMedianSortedArrays(arr_1, arr_2))

arr_1 = [3, 4, 4, 9, 11, 12, 13, 14]
arr_2 = [1, 4, 5]
print(Solution().findMedianSortedArrays(arr_1, arr_2))

arr_1 = [3, 4, 5, 9, 11, 12]
arr_2 = [1, 6]
print(Solution().findMedianSortedArrays(arr_1, arr_2))

arr_1 = [3, 4, 5, 9, 11, 13]
arr_2 = [12, 13]
print(Solution().findMedianSortedArrays(arr_1, arr_2))
