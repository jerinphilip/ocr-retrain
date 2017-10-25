from collections import defaultdict

class DSU:
    def __init__(self, **kwargs):
        self.V = kwargs['vertices']
        self.rank = [0 for _ in range(kwargs['vertices'])]
        self.parent = [i for i in range(kwargs['vertices'])]

    def merge(self, x, y):
        u = self.find(x)
        v = self.find(y)

        if self.rank[u] == self.rank[v]:
            self.rank[u] = self.rank[u] + 1
            self.parent[v] = u

        elif self.rank[u] > self.rank[v]:
            self.parent[v] = u

        else:
            self.parent[u] = v

    def find(self, x):
        if x == self.parent[x]: return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def same(self, x, y):
        return self.find(x) == self.find(y)




