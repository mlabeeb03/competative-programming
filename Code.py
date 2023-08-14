import sys

try:
    file = open("inp.txt", "r")
    input = lambda: file.readline().rstrip("\n\r")
except:
    input = lambda: sys.stdin.readline().rstrip("\n\r")

# _____________________________________________________________


def ini():
    return int(input())


def inl():
    return [i for i in input().split()]


def inli():
    return [int(i) for i in input().split()]


def ins():
    return [i for i in input()]


def insi():
    return [int(i) for i in input()]


# _____________________________________________________________

from collections import Counter, deque
from itertools import accumulate

# _____________________________________________________________

inf = float("inf")
mod = 1000000007

# _____________________________________________________________


def solve():
    r, c, mn, mx = inli()
    arr = [ins() for i in range(r)]
    plus = []
    ass = []
    for col in range(c):
        plus.append(0)
        ass.append(0)
        for row in range(r):
            if arr[row][col] == "+":
                ass[-1] += 1
            else:
                plus[-1] += 1
    plussum = list(accumulate(plus))
    assum = list(accumulate(ass))

    dpplus = [None] * c
    dpass = [None] * c
    for i in range(mn - 1, mx):
        dpplus[i] = plussum[i]
        dpass[i] = assum[i]

    # print(plussum)
    # print(assum)

    # print(dpplus)
    # print(dpass)

    for i in range(c):
        if dpplus[i] is not None:
            x = dpplus[i]
            # print(x)
            for j in range(i + mn, min(c, i + mx + 1)):
                if dpass[j] is None:
                    dpass[j] = x + assum[j] - assum[i + mn - 1]
                else:
                    dpass[j] = min(dpass[j], x + assum[j] - assum[i + mn - 1])

        if dpass[i] is not None:
            x = dpass[i]
            for j in range(i + mn, min(c, i + mx + 1)):
                if dpplus[j] is None:
                    dpplus[j] = x + plussum[j] - plussum[i + mn - 1]
                else:
                    dpplus[j] = min(dpplus[j], x + plussum[j] - plussum[i + mn - 1])
    if dpass[-1] is None:
        print(dpplus[-1])
    elif dpplus[-1] is None:
        print(dpass[-1])
    else:
        print(min(dpplus[-1], dpass[-1]))


for _ in range(1):
    solve()
