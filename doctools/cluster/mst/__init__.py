from .bijection import Bijection
from .graph import Graph

def cluster(X, d):
    n = len(X)
    f = Bijection()
    G = Graph()
    for i in range(n):
        for j in range(i+1, n):
            x, y = X[i], X[j]
            w = d(x, y)
            u, v = f(x), f(y)
            G.add_edge(u, v, w)

    clusters = G.cluster()


