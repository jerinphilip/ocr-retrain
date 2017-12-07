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


def recluster(G, T):
    edges = edges.keys()
    us, vs = list(zip(*edges))
    vertices = set(us).union(set(vs))
    n = len(vertices)
    NG = Graph(vertices=n)
    for i in range(n):
        for j in range(i+1, n):
            x, y = vertices[i], vertices[j]
            if (x, y) in G:
                w = G[(x, y)]
                NG.add_edge(i, j, w)

    options = {
        'components': lambda : G.cluster(**kwargs),
        'matrix': lambda : G.matrix(**kwargs)
    }
    if 'rep' in kwargs:
        return options.get(kwargs['rep'])()
    return G.matrix(**kwargs)

    
