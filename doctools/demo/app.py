from flask import Flask, render_template, jsonify, request
import doctools.parser.cluster as pc
from doctools.postproc.correction.params import cluster_params as params

app = Flask(__name__)

group_path = '/OCRData2/praveen-intermediate/group/'


def _d3(cluster, text):
    # Nodes is a list of vertices
    indices = set(text["errored"])
    predictions = text["predictions"]
    truths = text["truths"]
    nodes = []
    links = []

    def _ecount(cs):
        errored = lambda x: x in indices
        ls = list(filter(errored, cs))
        return len(ls)

    def connect(component, edges):
        _edges = {}
        n = len(component)
        # print(n)
        E = 0
        for i in range(n):
            for j in range(i+1, n):
                v, u = component[i], component[j]
                assert (not ((v, u) in edges and (u, v) in edges))
                if (v, u) in edges:
                    _edges[(v, u)] = edges[(v, u)]
                    E += 1
                if (u, v) in edges:
                    _edges[(u, v)] = edges[(u, v)]
                    E += 1

        for i in range(n):
            for j in range(i+1, n):
                if E < n-1:
                    v, u = component[i], component[j]
                    if (v, u) not in edges and (u, v) not in edges:
                        _edges[(v, u)] = 1
                        edges[(v, u)] = 1
                        E += 1
                else:
                    return _edges


    
    # Fix components, edges
    # Must have >= 4 errors.

    def errored(component):
        err = lambda i: i in indices
        ls = filter(err, component)
        return list(ls)

    threshold = 4
    components = cluster["components"]
    components = list(filter(lambda x: _ecount(x) >= threshold, components))
    components = list(map(errored, components))
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
    feat = 'ocr'
    text, status = pc.load(book, feat=feat)
    return jsonify(_d3(cluster, text))


if __name__ == '__main__':
    app.run('0.0.0.0', port=8080)

