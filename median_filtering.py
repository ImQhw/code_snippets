#!/usr/bin/env python3
from pprint import pprint
from typing import List

from quick_sort import quick_sort


def median_filtering(array2d: List[List[int]], kernel_size: int, sorter=quick_sort):
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

            sorted_neighbors = sorter(neighbors)

            array2d[y][x] = sorted_neighbors[mid_index]


def main():
    array2d = [
        [1, 2, 3],
        [4, 9, 6],
        [7, 5, 9],
    ]

    pprint("raw: ")
    pprint(array2d)

    median_filtering(array2d, 3, sorter=quick_sort)
    pprint("after: ")
    pprint(array2d)


if __name__ == "__main__":
    main()
