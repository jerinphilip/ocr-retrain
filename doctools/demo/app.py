from flask import Flask, render_template, jsonify
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
        n = len(component)
        print(n)
        E = 0
        for i in range(n):
            for j in range(i+1, n):
                v, u = component[i], component[j]
                if (v, u) in edges:
                    E += 1

        for i in range(n):
            for j in range(i+1, n):
                if E < n-1:
                    v, u = component[i], component[j]
                    edges[(v, u)] = 1
                    E += 1
                else:
                    return edges


    
    threshold = 4
    #components = sorted(cluster["components"], key=_ecount)
    components = cluster["components"]
    components = list(filter(lambda x: _ecount(x) >= threshold, components))
    edges = cluster["edges"]
    for component in components:
        edges = connect(component, edges)

    counter = 0
    d = {}
    for i, c in enumerate(components):
        for v in c:
            if v in indices:
                if not v in d:
                    d[v] = counter
                    counter = counter+1
                node = {"name": truths[v], "group": i}
                nodes.append(node)

    for e in edges:
        u, v = e
        if u in d and v in d:
            w = edges[e]
            w = int(100*w)
            link = {"source": d[u], "target": d[v], "value": w, "weight": w}
            links.append(link)


    return {"nodes": nodes, "links": links}


    # links contain edges.




@app.route('/<book>/')
def correct(book):
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

