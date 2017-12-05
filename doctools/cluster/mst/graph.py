from collections import defaultdict
from .dsu import DSU
import numpy as np

class Graph:
    def __init__(self, **kwargs):
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
        edges = self.tree().items()
        adj = defaultdict(list)
        T = kwargs['threshold']
        new_edges = {}
        for edge in edges:
            link, w = edge
            if w < T:
                u, v = link
                adj[u].append(v)
                adj[v].append(u)
                new_edges[link] = w

        visited = dict([(u, 0) for u in adj])
        # Do DFS on adj to get components
        def dfs(u):
            traversed = []
            stack = [u]
            while stack:
                _u = stack.pop()
                visited[_u] = 2
                traversed.append(_u)
                for v in adj[_u]:
                    if visited[v] == 0:
                        visited[v] = 1
                        stack.append(v)
            return traversed

        components = []
            
        for u in adj:
            if not visited[u]:
                connected = dfs(u)
                components.append(connected)

        # Post computation assertions
        included = set()
        for component in components:
            cset = set(component)
            assert(not (cset.intersection(included)))
            included.union(cset)



        return (new_edges, components)

    def matrix(self, **kwargs):
        edges = self.tree().items()
        T = kwargs['threshold']
        adj = np.zeros((self.V, self.V))
        for edge in edges:
            link, w = edge
            if w < T:
                u, v = link
                adj[u][v] = w
                adj[v][u] = w

        return adj



