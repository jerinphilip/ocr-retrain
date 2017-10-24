from collections import defaultdict

class DSU:
    def __init__(self, **kwargs):
        self.V = kwargs['vertices']
        self.rank = [0 for _ in range(kwargs['vertices'])]
        self.parent = [i for i in range(kwargs['vertices'])]

    def merge(self, x, y):
        u = self.find(x)
        v = self.find(y)

        if R[u] == R[v]:
            R[u] = R[u] + 1
            parent[v] = u

        elif R[u] > R[v]:
            parent[v] = u

        else:
            parent[u] = v

    def find(self, x):
        if x == parent[x]: return x
        parent[x] = self._find(parent[x])
        return parent[x]

    def same(self, x, y):
        return D.find(x) == D.find(y)




