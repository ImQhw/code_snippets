#!/usr/bin/env python3

"""
From "Programming Pearls, Second Edition, Jon Bentley", chapter 9
"""


def binary_search(a: list, t: int) -> int:
    low, high = 0, len(a) - 1

    while low < high:
        # invariant a[low] <= t <= a[high]
        mid = low + (high - low) // 2
        cmp = t - a[mid]

        if cmp == 0:
            return mid

        # a[low] <= t < a[mid]
        if 0 < cmp:
            high = mid - 1  # keep invariant a[low] <= t <= a[high=mid-1]

        # a[mid] < t < a[high]
        else:
            low = mid + 1  # keep invariant a[low=mid+1] <= t <= a[high]

    return -1


def binary_search_first(a: list, t: int) -> int:
    # assumes that a[-1] < t <= a[n]
    # but we will never access a[-1] and a[n]
    low, high = -1, len(a)

    # when we find an answer, it must satisfy
    # a[low] < t <= a[high], so (low + 1) == high
    while (low + 1) != high:
        # invariant a[low] < t <= a[high]
        mid = low + (high - low) // 2

        # a[mid] < t <= a[high]
        if a[mid] < t:
            low = mid  # keep invariant a[low=mid] < t <= a[high]

        # a[low] < t <= a[mid]
        else:
            high = mid  # keep invariant a[low] < t <= a[high=mid]

    return -1 if (high >= len(a) or a[high] != t) else high


def binary_search_last(a: list, t: int) -> int:
    # assumes that a[-1] <= t < a[n]
    # but we will never access a[-1] and a[n]
    low, high = -1, len(a)

    # when we find an answer, it must satisfy
    # a[low] <= t < a[high], so (low + 1) == high
    while (low + 1) != high:
        # invariant a[low] <= t < a[high]
        mid = low + (high - low) // 2

        # a[mid] <= t < a[high]
        if a[mid] <= t:
            low = mid  # keep invariant a[low=mid] <= t < a[high]

        # a[low] <= t < a[mid]
        else:
            high = mid  # keep invariant a[low] <= t < a[high=mid]

    return -1 if (low <= -1 or a[low] != t) else low


def main():
    assert binary_search_first([1], 1) == 0
    assert binary_search_first([1, 1], 1) == 0
    assert binary_search_first([1, 1, 2], 2) == 2
    assert binary_search_first([1, 1, 2, 2], 2) == 2
    assert binary_search_last([1, 1, 2, 2], 2) == 3
    assert binary_search_last([1, 1, 2, 2], 1) == 1
    assert binary_search_last([1, 1, 2, 2], 0) == -1
    assert binary_search_first([1, 1, 2, 2], 0) == -1
    assert binary_search([1, 1, 2, 2], 0) == -1
    assert binary_search([1, 1, 2, 2], 1) == 1


if __name__ == "__main__":
    main()
