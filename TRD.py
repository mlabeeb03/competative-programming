# fmt: off
# ---------- MISC ----------


# PREVENT STACK OVERFLOW
# put below snippet in your code and then put @bootstrap over recursive function
# replace all instances of return with yield
# if a function does not return anything still write yield at the end
# if you are calling another function from within, do it with yield

from types import GeneratorType
def bootstrap(f, stack=[]):
    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to
    return wrappedfunc
# example
@bootstrap
def dfs(visited, graph, node, arr, brr, ans):
    if node not in visited:
        visited.add(node)
        for neighbour in graph[node]:
            yield dfs(visited, graph, neighbour[0], arr, brr, ans)
    arr.pop()
    brr.pop()
    yield


# ---------- GRAPH ----------


# UNION FIND(DSU)
def find(v):
    if v == parent[v]:
        return v
    parent[v] = find(parent[v])
    return parent[v]
def union(a, b):
    a, b = find(a), find(b)
    if a != b:
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
n = 5
parent, size = [i for i in range(n + 1)], [1 for i in range(n + 1)]


# DFS
def dfs(g, i):
    stack = [i]
    visited = set()
    while stack:
        start = stack[-1]
        if start not in visited:
            visited.add(start)
            for child in g[start]:
                stack.append(child)
        else:
            stack.pop()


# BFS
from collections import deque
def bfs(g, i):
    stack = deque([i])
    visited = set()
    while stack:
        start = stack[0]
        if start not in visited:
            visited.add(start)
            for child in g[start]:
                stack.append(child)
        else:
            stack.popleft()


# CHECK IF A GRAPH IS BIPARTITE
def isBipartite(g):
	col = [-1]*(len(g))
	q = []
	for i in range(len(g)):
		if (col[i] == -1):
			q.append([i, 0])
			col[i] = 0		
			while len(q) != 0:
				p = q[0]
				q.pop(0)
				v = p[0]
				c = p[1]
				for j in g[v]:
					if (col[j] == c):
						return False
					if (col[j] == -1):
						if c == 1:
							col[j] = 0
						else:
							col[j] = 1
						q.append([j, col[j]])
	return True

g = [[1, 2], [0], [0, 3], [2]]
print(isBipartite(g))



def bipartiteMatch(graph):
	'''Find maximum cardinality matching of a bipartite graph (U,V,E).
	The input format is a dictionary mapping members of U to a list
	of their neighbors in V.  The output is a triple (M,A,B) where M is a
	dictionary mapping members of V to their matches in U, A is the part
	of the maximum independent set in U, and B is the part of the MIS in V.
	The same object may occur in both U and V, and is treated as two
	distinct vertices if this happens.'''
	
	# initialize greedy matching (redundant, but faster than full search)
	matching = {}
	for u in graph:
		for v in graph[u]:
			if v not in matching:
				matching[v] = u
				break
	while 1:
		# structure residual graph into layers
		# pred[u] gives the neighbor in the previous layer for u in U
		# preds[v] gives a list of neighbors in the previous layer for v in V
		# unmatched gives a list of unmatched vertices in final layer of V,
		# and is also used as a flag value for pred[u] when u is in the first layer
		preds = {}
		unmatched = []
		pred = dict([(u,unmatched) for u in graph])
		for v in matching:
			del pred[matching[v]]
		layer = list(pred)
		
		# repeatedly extend layering structure by another pair of layers
		while layer and not unmatched:
			newLayer = {}
			for u in layer:
				for v in graph[u]:
					if v not in preds:
						newLayer.setdefault(v,[]).append(u)
			layer = []
			for v in newLayer:
				preds[v] = newLayer[v]
				if v in matching:
					layer.append(matching[v])
					pred[matching[v]] = v
				else:
					unmatched.append(v)
		
		# did we finish layering without finding any alternating paths?
		if not unmatched:
			unlayered = {}
			for u in graph:
				for v in graph[u]:
					if v not in preds:
						unlayered[v] = None
			return (matching,list(pred),list(unlayered))

		# recursively search backward through layers to find alternating paths
		# recursion returns true if found path, false otherwise
		def recurse(v):
			if v in preds:
				L = preds[v]
				del preds[v]
				for u in L:
					if u in pred:
						pu = pred[u]
						del pred[u]
						if pu is unmatched or recurse(pu):
							matching[v] = u
							return 1
			return 0

		for v in unmatched: recurse(v)

# https://github.com/johnjdc/minimum-vertex-cover/blob/master/MVC.py
# Find a minimum vertex cover
def min_vertex_cover(left_v, right_v):
    '''Use the Hopcroft-Karp algorithm to find a maximum
    matching or maximum independent set of a bipartite graph.
    Next, find a minimum vertex cover by finding the 
    complement of a maximum independent set.
    The function takes as input two dictionaries, one for the
    left vertices and one for the right vertices. Each key in 
    the left dictionary is a left vertex with a value equal to 
    a list of the right vertices that are connected to the key 
    by an edge. The right dictionary is structured similarly.
    The output is a dictionary with keys equal to the vertices
    in a minimum vertex cover and values equal to lists of the 
    vertices connected to the key by an edge.
    For example, using the following simple bipartite graph:
    1000 2000
    1001 2000
    where vertices 1000 and 1001 each have one edge and 2000 has 
    two edges, the input would be:
    left = {1000: [2000], 1001: [2000]}
    right = {2000: [1000, 1001]}
    and the ouput or minimum vertex cover would be:
    {2000: [1000, 1001]}
    with vertex 2000 being the minimum vertex cover.
    The code can also generate a bipartite graph with an arbitrary
    number of edges and vertices, write the graph to a file, and 
    read the graph and convert it to the appropriate format.'''



    data_hk = bipartiteMatch(left_v)
    print(data_hk)
    left_mis = data_hk[1]
    right_mis = data_hk[2]
    mvc = left_v.copy()
    mvc.update(right_v)

    for v in left_mis:
        del(mvc[v])
    for v in right_mis:
        del(mvc[v])

    return mvc

leftv = {'a': [1, 3], 'c': [1, 3], 'd': [3, 6], 'h': [8], 'i': [8]}
rightv = {1: ['a', 'c'], 3: ['a', 'c', 'd'], 6: ['d'], 8: ['h', 'i']}
print(min_vertex_cover(leftv, rightv))


# ---------- BITS ----------


# bits required to store an integer
print((999999).bit_length())

# 018 formats the number to eighteen digits zero-padded on the left
# b converts the number to its binary representation
print(format(8, "018b"))  #'00000110'


def ConvertDecimalToBaseX(num, x):
    if num == 0:
        return [0]
    digits = []
    while num:
        digits.append(int(num % x))
        num //= x
    return digits[::-1]


def ConvertBaseXToDecimal(num, CurrBase):
    ans = 0
    for i in map(int, num):
        ans = CurrBase * ans + i
    return ans


# ---------- NUMBER THEORY ----------



def SieveOfEratosthenes(num):
    prime = [True for i in range(num + 1)]
    p = 2
    while p * p <= num:
        if prime[p] == True:
            for i in range(p * p, num + 1, p):
                prime[i] = False
        p += 1
    for p in range(2, num + 1):
        if prime[p]:
            PRIMES.add(p * p)
PRIMES = set()


def fastSieve(n):
    r = [False, True] * (n // 2) + [True]
    r[1], r[2] = False, True
    for i in range(3, int(1 + n**0.5), 2):
        if r[i]:
            r[i * i :: 2 * i] = [False] * ((n + 2 * i - 1 - i * i) // (2 * i))
    return r


from functools import reduce
def factors(n):
    return list(
        set(reduce(list.__add__,([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),)))


def is_square(apositiveint):
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen:
            return False
        seen.add(x)
    return True


def gcd(x, y):
    while y:
        x, y = y, x % y
    return x


def lcm(x, y):
    return (x * y) // gcd(x, y)


# ---------- MATRIX ----------


def rotate_a_matrix(a) :
	n = len(a)
	for r in range(n//2):
		for c in range(r, n - r - 1):
			a[r][c], a[c][n-1-r], a[n-1-r][n-1-c], a[n-1-c][r] = a[n-1-c][r], a[r][c], a[c][n-1-r], a[n-1-r][n-1-c] # for clockwise
			a[r][c], a[c][n-1-r], a[n-1-r][n-1-c], a[n-1-c][r] = a[c][n-1-r], a[n-1-r][n-1-c], a[n-1-c][r], a[r][c] # for anti clockwise

# Largest sum submatrix
def kadane(arr, start, finish, n):
	Sum = 0
	maxSum = -999999999999
	i = None
	finish[0] = -1
	local_start = 0
	for i in range(n):
		Sum += arr[i]
		if Sum < 0:
			Sum = 0
			local_start = i + 1
		elif Sum > maxSum:
			maxSum = Sum
			start[0] = local_start
			finish[0] = i
	if finish[0] != -1:
		return maxSum
	maxSum = arr[0]
	start[0] = finish[0] = 0
	for i in range(1, n):
		if arr[i] > maxSum:
			maxSum = arr[i]
			start[0] = finish[0] = i
	return maxSum

def findMaxSum(M):
	global ROW, COL
	maxSum, finalLeft = -999999999999, None
	finalRight, finalTop, finalBottom = None, None, None
	left, right, i = None, None, None

	temp = [None] * ROW
	Sum = 0
	start = [0]
	finish = [0]
	for left in range(COL):
		temp = [0] * ROW
		for right in range(left, COL):
			for i in range(ROW):
				temp[i] += M[i][right]
			Sum = kadane(temp, start, finish, ROW)
			if Sum > maxSum:
				maxSum = Sum
				finalLeft = left
				finalRight = right
				finalTop = start[0]
				finalBottom = finish[0]
	print("(Top, Left)", "(", finalTop,
		finalLeft, ")")
	print("(Bottom, Right)", "(", finalBottom,
		finalRight, ")")
	print("Max sum is:", maxSum)

ROW = 4
COL = 5
M = [[1, 2, -1, -4, -20],
	[-8, -3, 4, 2, 1],
	[3, 8, 10, 1, 3],
	[-4, -1, 1, 7, -6]]

# Flood Fill
n = rows = 10
m = columns = 10
a = [] # input array
q = []
four = [(1, 0), (-1, 0), (0, 1), (0, -1)]
eight = [(-1, -1), (-1, 0), (-1, 1),(0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
for i in range(1, n):
    for j in range(1, m):
        if a[i][j] == '.':
            q.append((i, j))
            while q:
                x, y = q.pop()           
                a[x][y] = '#'                
                for dx, dy in four:
                    if a[x + dx][y + dy] == '.':
                        q.append((x + dx, y + dy))


# ---------- SUBARRAYS ----------


def find_palindromic_substrings(s):  # manacher's algorithm
    t = "#" + "#".join(s) + "#"
    n = len(t)
    p = [0] * n
    c = 0
    r = 0
    for i in range(n):
        if i < r:
            p[i] = min(r - i, p[2 * c - i])
        else:
            p[i] = 0
        while (
            i - p[i] - 1 >= 0
            and i + p[i] + 1 < n
            and t[i - p[i] - 1] == t[i + p[i] + 1]
        ):
            p[i] += 1
        if i + p[i] > r:
            c = i
            r = i + p[i]
    palindromes = []
    for i in range(n):
        if p[i] > 0:
            start = (i - p[i]) // 2
            end = start + p[i]
            palindromes.append(s[start:end])
    return palindromes


def count_subarrays_with_given_xor(a, target):
    n = len(a)
    occ = [0] * (100001)  # num of times each number occured in xor prefix arrays
    occ[0] = 1  # 0 occured once in empty subarray
    c = 0  # count for subarrays with xor equal to target
    x = 0  # prefix xor uptill i
    for i in range(n):
        x ^= a[i]
        c += occ[target ^ x]
        occ[x] += 1
    return c

def sqrt_decomposition(arr, bin):
    n = len(arr)
    segment_size = int(n**0.5) + 1
    xor_segments = []
    zero_cor = []
    one_xor = []
    for i in range(n):
        if i % segment_size == 0:
            xor_segments.append([])
            zero_cor.append(0)
            one_xor.append(0)
        xor_segments[-1].append(arr[i])
        if bin[i] == 0:
            zero_cor[-1] ^= arr[i]
        else:
            one_xor[-1] ^= arr[i]
    return xor_segments, zero_cor, one_xor

def update(n, arr, zero, one, bin, l, r):
    segment_size = int(n**0.5) + 1
    rightmost_segment = r // segment_size
    while l <= r:
        segment_number = l // segment_size
        position_within_segment = l % segment_size

        if position_within_segment == 0 and segment_number < rightmost_segment:
            zero[segment_number], one[segment_number] = (
                one[segment_number],
                zero[segment_number],
            )
            next_segment_start = (segment_number + 1) * segment_size
            l = next_segment_start
        else:
            zero[segment_number] ^= arr[segment_number][position_within_segment]
            one[segment_number] ^= arr[segment_number][position_within_segment]
            l += 1

def KMP(w, t):
    if w == "":
        return 0
    lps = [0] * len(w)
    m, i = 0, 1
    while i < len(w):
        if w[m] == w[i]:
            lps[i] = m + 1
            m += 1
            i += 1
        else:
            if m == 0:
                lps[i] = 0
                i += 1
            else:
                m = lps[m - 1]
    
    i = 0 # for t
    j = 0 # for w
    occurrences = []
    while i < len(t):
        if w[j] == t[i]:
            i, j = i + 1, j + 1
        else:
            if j == 0:
                i += 1
            else:
                j = lps[j - 1]
        if j == len(w):
            occurrences.append(i - len(w))
            j = lps[j - 1]
    return occurrences


# ---------- SUBSTRINGS ----------


# Longest increasing subsequence in nlgn
from bisect import bisect_left
class Solution:
    def lengthOfLIS(nums):
        ans = [nums[0]]
        for i in range(1, len(nums)):
            if nums[i] > ans[-1]:
                ans.append(nums[i])
            else:
                ind = bisect_left(ans, nums[i])
                ans[ind] = nums[i]
        return len(ans)
    

# ---------- SEGMENT TREE ----------


from math import inf
n = int(input())
a = list(map(int, input().split()))
t = [inf] * (4 * n)
# initial array, current vertex(1), array left(0), array right(n - 1)
def build(a, v, tl, tr):
    if tl == tr:
        t[v] = a[tl]
    else:
        tm = (tl + tr) // 2
        build(a, v * 2, tl, tm)
        build(a, v * 2 + 1, tm + 1, tr)
        t[v] = min(t[v * 2], t[v * 2 + 1])
# current vertex(1), array left(0), array right(n - 1), query left, query right (0 indexed)
def query(v, tl, tr, l, r):
    if l > r:
        return inf
    if l == tl and r == tr:
        return t[v]
    tm = (tl + tr) // 2
    return min(
        query(v * 2, tl, tm, l, min(r, tm)),
        query(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r),
    )
# current vertex(1), array left(0), array right(n - 1), position (0 indexed), new value
def update(v, tl, tr, pos, new_val):
    if tl == tr:
        t[v] = new_val
    else:
        tm = (tl + tr) // 2
        if pos <= tm:
            update(v * 2, tl, tm, pos, new_val)
        else:
            update(v * 2 + 1, tm + 1, tr, pos, new_val)
        t[v] = min(t[v * 2], t[v * 2 + 1])
build(a, 1, 0, n - 1)
