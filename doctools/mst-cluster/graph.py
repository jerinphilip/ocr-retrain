
class Graph:
    def __init__(**kwargs):
        self.edges = {}

    def add_edge(self, u, v, w):
        _u, _v = min(u, v), max(u, v)
        self.edges[(_u, _v)] = w

    def cluster(self):
        """ Define clustering here """
        orderedEdges = sorted(self.edges.items(), key = lambda x: x[1])
        for edge in orderedEdges:
            print(edge)
