
def as_dot(words, edges, component):
    def uid(i):
        return 'node_{}'.format(i)
        #return '{}_{}'.format(i, i)

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

def old_as_dot(words, edges, components):
    def uid(i):
        return 'node_{}'.format(i)
        #return '{}_{}'.format(i, i)

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
    clusters = []
    for c, component in enumerate(components):
        clusters.append(clusterstr(c, component))

    clusters_r = '\n'.join(clusters)
    _edges = []
    for _edge in edges:
        _edges.append(edge(*_edge))

    edgestr = '\n'.join(_edges)

    return '{} {{\n {} \n {}\n}}'.format(decl, clusters_r, edgestr)

def as_dict(words, components, d):
    nodes = []
    links = []
    for c, component in enumerate(components):
        n = len(component)
        for i in range(n):
            nodes.append({
                "id": "{}-{}-{}".format(words[i], c, i), 
                "group": c
            })

            for j in range(i+1, n):
                src = words[i]
                tgt = words[j]
                links.append({
                    "source": "{}-{}-{}".format(words[i], c, i),
                    "target": "{}-{}-{}".format(words[j], c, j),
                    "value": d(src, tgt)
                    
                })
    return {
        "nodes": nodes,
        "links": links
    }




