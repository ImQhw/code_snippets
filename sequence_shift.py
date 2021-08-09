#!/usr/bin/env python3

"""
From "Programming Pearls, Second Edition, Jon Bentley", chapter 2.3
"""


def reverse(sequence: list, start, stop):
    """reverse sequence[start...stop](exclude stop)"""

    i, j = start, stop - 1

    while i < j:
        sequence[i], sequence[j] = sequence[j], sequence[i]
        i += 1
        j -= 1


def shift(sequence: list, offset: int):
    if offset == 0:
        return

    n = len(sequence)

    # convert right-shift to left-shift
    if offset < 0:
        offset = n + offset

    # A|B --> inv_B|inv_A
    reverse(sequence, 0, n)

    # inv_B --> B
    reverse(sequence, 0, n - offset)

    # inv_A -- > A
    reverse(sequence, n - offset, n)


def main():
    src = list("abcdefgh")
    shift(src, 3)

    assert "".join(src) == "defghabc", src

    src = list("abcdefgh")
    shift(src, 4)

    assert "".join(src) == "efghabcd", src

    src = list("abcdefgh")
    shift(src, 0)

    assert "".join(src) == "abcdefgh", src

    src = list("abcdefgh")
    shift(src, -1)

    assert "".join(src) == "habcdefg", src


if __name__ == "__main__":
    main()
