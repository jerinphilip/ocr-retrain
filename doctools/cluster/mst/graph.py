from .dsu import DSU

class Graph:
    def __init__(**kwargs):
        self.V = kwargs['vertices']
        self.edges = {}

    def add_edge(self, u, v, w):
        _u, _v = min(u, v), max(u, v)
        self.edges[(_u, _v)] = w

    def tree(self):
        D = DSU(vertices=self.V)
        orderedEdges = sorted(self.edges.items(), key = lambda x: x[1])
        nodeCount = 0
        edges = {}
        for edge in orderedEdges:
            (u, v), w = edge

            if not D.same(u, v):
                edges[(u,v)] = w
                nodeCount += 1
                D.merge(u, v)

            if nodeCount == self.V - 1:
                break

        return edges

    def cluster(self, **kwargs):
        """ Define clustering here """
        edges = self.tree()
        disconnected = {}
        T = kwargs['threshold']
        for edge in edges:
            link, w = edge
            if w < T:
                break
            disconnected[link] = w

