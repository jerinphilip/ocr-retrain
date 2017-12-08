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
            #u, v = f(x), f(y)
            if w < kwargs['prune_above']:
                G.add_edge(i, j, w)

    options = {
        'components': lambda : G.cluster(**kwargs),
        'matrix': lambda : G.matrix(**kwargs)
    }
    if 'rep' in kwargs:
        return options.get(kwargs['rep'])()
    return G.matrix(**kwargs)


def merge(E1, E2, X):
    n = len(X)
    f = Bijection()
    G = Graph(vertices=len(X))
    for link, weight in E1.items():
            
        u1, v1 = link
        G.add_edge(u1, v1, weight)
    for link, weight in E2.items():
        u2, v2 = link
        G.add_edge(u2, v2, weight)

    return G.cluster(threshold=1)

def recluster(G, n, **kwargs):
    edges = G.keys()
    us, vs = list(zip(*edges))
    NG = Graph(vertices=n)
    for (x, y) in G:
        w = G[(x, y)]
        NG.add_edge(x, y, w)

    options = {
        'components': lambda : NG.cluster(**kwargs),
        'matrix': lambda : NG.matrix(**kwargs)
    }
    if 'rep' in kwargs:
        return options.get(kwargs['rep'])()
    return NG.matrix(**kwargs)

    
