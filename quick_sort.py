#!/usr/bin/env python3
import random
from pprint import pprint
from typing import List


def partition(a: List[int], low, high) -> int:
    # use a[low] as pivot value
    mid = low

    for i in range(low + 1, high + 1):
        # invariant: a[low+1...mid] < a[low] <= a[mid+1...i]
        if a[i] < a[low]:
            mid += 1
            a[mid], a[i] = a[i], a[mid]

    # before swap, a[mid] < a[low], and after swap,
    # it make a[low...mid-1] < a[mid] <= a[mid+1...high]
    a[mid], a[low] = a[low], a[mid]

    return mid


def partition2(a: List[int], low, high) -> int:
    pivot_value = a[low]

    i, j = low + 1, high
    while True:
        # invariant: a[i] <= pivot_value <= a[j]

        # skip a[i] < pivot_value
        while i <= j and a[i] < pivot_value:
            i += 1

        # skip pivot_value < a[j]
        while pivot_value < a[j]:
            j -= 1

        if i >= j:
            break

        # a[j] <= pivot_value <= a[i]
        a[i], a[j] = a[j], a[i]
        # a[i] <= pivot_value <= a[j]

    # when break, i >= j
    # to keep a[low...j-1] < a[j] < a[j+1...high]
    # we should swap a[low] with a[j], not a[i]
    # and return j as pivot
    a[low], a[j] = a[j], a[low]

    return j


def quick_sort(a: List[int]) -> List[int]:
    def _quick_sort(low, high):
        if low >= high:  # process single element or empty array
            return

        j = partition2(a, low, high)

        _quick_sort(low, j - 1)
        _quick_sort(j + 1, high)

    # shuffle input, make sort performance independent to input distribution
    random.shuffle(a)

    _quick_sort(0, len(a) - 1)

    return a


def main():

    a = [i for i in range(10)] + [i for i in range(10)]
    random.shuffle(a)

    pprint(a)
    pprint(sorted(a))

    pprint(a)
    pprint(quick_sort(a))


if __name__ == "__main__":
    main()
