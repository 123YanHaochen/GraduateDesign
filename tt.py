import sys
from collections import Counter
from functools import reduce
from heapq import nsmallest
from math import inf
from itertools import accumulate, permutations
import time
input = lambda: sys.stdin.readline().strip()

def I():
    return input()
def II():
    return int(input())
def MII():
    return map(int, input().split())
def LI():
    return list(input().split())
def LII():
    return list(map(int, input().split()))
def LFI():
    return list(map(float, input().split()))
def GMI():
    return map(lambda x: int(x) - 1, input().split())
def LGMI():
    return list(map(lambda x: int(x) - 1, input().split()))

'''
d[0] = 0
d[i] = nums[i] - nums[i-1]
[l, r]增加1 相当于 d[r+1] -= 1 d[l] += 1
1. 考虑最终结果 对应差分数组全部为0
2.两种操作映射到差分数组上 [0, i] 相当于d[0] -= 1 d[i+1] += 1 (i>=0) 也就是d[0] -= 1 d[j] += 1 (j > 0)
            [i, n - 1] 相当于 d[i] -= 1 d[n] += 1(超出范围) 相当于 任意选择一个数字-1
'''
def solve():
    n = II()
    nums = LII()
    d = [0] * n
    d[0] = nums[0]
    for i in range(1, n):
        d[i] = nums[i] - nums[i-1]
    # print(d)
    for i in range(1, n):
        if d[i] == 0: continue
        if d[i] < 0:
            d[0] += d[i]
            d[i] = 0
            if d[0] < 0:
                print("NO")
                return
    print("YES")

    return

t = II()
# t = 1
for _ in range(t):
    try:
        solve()
    except Exception as e:
        print(e)
