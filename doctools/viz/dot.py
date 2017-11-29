def as_dot(words, edges, component):
    def uid(i):
        return 'node_{}'.format(i)

    def node(i):
        return '{} [label="{}"];'.format(uid(i), words[i])
    
    def edge(i, j):
        return '{} -- {};'.format(uid(i), uid(j))

    def clusterstr(i, component):
        decl = 'subgraph cluster_{}'.format(i)
        nodes = list(map(node, component))
        nodestr = '\n'.join(nodes)
        return "{} {{\n {} \n}}".format(decl, nodestr)


    decl = 'graph G'
    nodes = list(map(node, component))
    nodestr = '\n'.join(nodes)
    _edges = []
    for _edge in edges:
        u, v = _edge
        if u in component and v in component:
            _edges.append(edge(*_edge))

    edgestr = '\n'.join(_edges)

    return '{} {{\n {} \n {}\n}}'.format(decl, nodestr, edgestr)

