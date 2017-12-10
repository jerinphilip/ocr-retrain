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
            if w < kwargs['prune_above']:
                G.add_edge(i, j, w)

    options = {
        'components': lambda : G.cluster(**kwargs),
        'matrix': lambda : G.matrix(**kwargs)
    }
    if 'rep' in kwargs:
        return options.get(kwargs['rep'])()
    return G.matrix(**kwargs)


def merge(E1, E2, n):
    f = Bijection()
    G = Graph(vertices=n)
    for link, weight in E1.items():
            
        u1, v1 = link
        G.add_edge(u1, v1, weight)
    for link, weight in E2.items():
        u2, v2 = link
        G.add_edge(u2, v2, weight)

    return G.cluster(threshold=1)

def intersection(E1, E2, n):
    f = Bijection()
    G = Graph(vertices=n)
    for e1 in E1:
        if e1 in E2:
            u, v = e1
            w = 1.0
            # print(u, v, w)
            G.add_edge(u, v, w)

    return G.cluster(threshold=2.0)

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

    

def bugfixcomponents(components, vertices):
    cs = components
    full = set(range(vertices))
    for component in components:
        full = full - set(component)
    for v in full:
        cs.append([v])
    return cs

