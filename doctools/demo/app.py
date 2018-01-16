from flask import Flask, render_template, jsonify, request
import doctools.parser.cluster as pc
from doctools.postproc.correction.params import cluster_params as params
from doctools.cluster.mst import recluster
import random
from collections import defaultdict

app = Flask(__name__)

group_path = '/OCRData2/praveen-intermediate/group/'


def connect(component, edges):
    _edges = {}
    n = len(component)
    adj = defaultdict(list)
    for i in range(n):
        for j in range(i, n):
            u, v = component[i], component[j]
            if (u, v) in edges:
                mu, mv = min(u, v), max(u, v)
                _edges[(mu, mv)] = edges[(u, v)]
                adj[u].append(v)
                adj[v].append(u)
    # Not everything would me connected.
    # Magic should convert this into a tree, preserving original edges.
    visited = dict([(u, 0) for u in component])
    subcomponents = []

    def dfs(_u):
        stack = [_u]
        traversed = []
        while stack:
            u = stack.pop()
            traversed.append(u)
            visited[u] = 2
            for v in adj[u]:
                if visited[v] == 0:
                    visited[v] = 1
                    stack.append(v)
        return traversed


    for u in component:
        if not visited[u]:
            subcomponents.append(dfs(u))

    
    for i in range(len(subcomponents)-1):
        cu = subcomponents[i]
        cv = subcomponents[i+1]

        # Connect cu and cv

        _u, _v = random.choice(cu), random.choice(cv)
        u, v = min(_u, _v), max(_u, _v)
        _edges[(u, v)] = 1
    return _edges



def _d3(cluster, text):
    # Nodes is a list of vertices
    indices = set(text["errored"])
    predictions = text["predictions"]
    truths = text["truths"]
    nodes = []
    links = []
    lbt, ubt = 2, 50
    def _ecount(cs):
        errored = lambda x: x in indices
        ls = list(filter(errored, cs))
        return len(ls)
    def errored(component):
        err = lambda i: i in indices
        ls = filter(err, component)
        return list(ls)
    components = cluster["components"]
    components = list(filter(lambda x: _ecount(x) >= lbt, components))
    components = list(map(errored, components))
    samples = min(len(components), 10) 
    components = random.sample(components, samples)
    edges = cluster["edges"]
    _edges = {}

    for component in components:
        es = connect(component, edges)
        print("Components:", len(component), "Edges:", len(es))
        _edges.update(es)

    # Generate integer labels
    counter = 0
    d = {}
    for component in components:
        for v in component:
            if not v in d:
                d[v] = counter
                counter = counter+1

    for i, c in enumerate(components):
        for v in c:
            label = '{}/{}'.format(predictions[v], truths[v])
            node = {"name": label, "group": i}
            nodes.append(node)

    for e in _edges:
        u, v = e
        w = _edges[e]
        w = int(100*w)
        link = {"source": d[u], "target": d[v], "value": w, "weight": w}
        links.append(link)


    return {"nodes": nodes, "links": links}


    # links contain edges.




@app.route('/<book>/', methods=['GET'])
def correct(book):
    if request.args.get('rep') == 'tree':
        return render_template('tree.html')
    return render_template('main.html')


@app.route('/<book>/graph')
def graph(book):
    feat = 'images'
    cluster, status = pc.load(book, feat=feat, **params[feat])
    e, c = recluster(cluster["edges"], cluster["vertices"], threshold=0.15
            , rep = 'components')
    print(e, c)
    cluster = {
        "edges": e,
        "components": c,
        "vertices": cluster["vertices"]
    }
    feat = 'ocr'
    text, status = pc.load(book, feat=feat)
    return jsonify(_d3(cluster, text))


if __name__ == '__main__':
    app.run('0.0.0.0', port=8080)

