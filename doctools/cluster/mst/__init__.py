from .bijection import Bijection
from .graph import Graph

def cluster(X, d, **kwargs):
    n = len(X)
    f = Bijection()
    G = Graph(vertices=len(X))
    for i in range(n):
        for j in range(i+1, n):
            x, y = X[i], X[j]
            w = d(x, y)
            u, v = f(x), f(y)
            G.add_edge(u, v, w)
            #print("Adding", u, v)
            #print("Adding", x, y)

    # clusters = G.cluster()
    matrix = G.matrix(**kwargs)
    return matrix


