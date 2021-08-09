#!/usr/bin/env python3
from pprint import pprint
from typing import List


def insertion_sort(a: List[int]) -> List[int]:
    if len(a) <= 1:
        return a

    n = len(a)

    for i in range(1, n):
        # invariant: a[0...i-1] sorted before insertion
        # and a[0...i] sorted after insertion

        insert_value = a[i]

        j = i
        while j > 0 and a[j - 1] > insert_value:
            # move a[j-1] --> a[j]
            a[j] = a[j - 1]
            j -= 1

        # a[j-1] <= insert_value
        a[j] = insert_value

    return a


def median_filtering(array2d: List[List[int]], kernel_size: int):
    assert kernel_size & 1, "kernel_size must be odd!"
    assert array2d, "empty array!"

    half_k = kernel_size // 2

    def extract_neighbors(cx, cy) -> List[int]:
        _neighbors = []

        for dy in range(-half_k, half_k + 1):
            for dx in range(-half_k, half_k + 1):
                _neighbors.append(array2d[cy + dy][cx + dx])

        return _neighbors

    mid_index = (kernel_size * kernel_size) // 2

    H, W = len(array2d), len(array2d[0])

    for y in range(half_k, H - half_k):
        for x in range(half_k, W - half_k):
            neighbors = extract_neighbors(x, y)

            sorted_neighbors = insertion_sort(neighbors)

            array2d[y][x] = sorted_neighbors[mid_index]


def main():
    array2d = [
        [1, 2, 3],
        [4, 9, 6],
        [7, 5, 9],
    ]

    pprint("raw: ")
    pprint(array2d)

    median_filtering(array2d, 3)
    pprint("after: ")
    pprint(array2d)

    print(insertion_sort([2, 3, 1]))


if __name__ == "__main__":
    main()
