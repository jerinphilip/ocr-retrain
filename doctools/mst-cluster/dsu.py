
class DSU:
    def __init__(self, **kwargs):
        self.rank = [0 for _ in range(kwargs['vertices'])]
        self.parent = [i for i in range(kwargs['vertices'])]

    def _merge(self, x, y):
        u = self.find(x)
        v = self.find(y)

        if R[u] == R[v]:
            R[u] = R[u] + 1
            parent[v] = u

        elif R[u] > R[v]:
            parent[v] = u

        else:
            parent[u] = v

    def _find(self, x):
        if x == parent[x]: return x
        parent[x] = self._find(parent[x])
        return parent[x]

    def find(self, x):
        return self._find(x-1)

    def merge(self, x, y):
        self._merge(x-1, y-1):w




