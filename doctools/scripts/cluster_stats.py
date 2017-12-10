import doctools.parser.cluster as pc
from doctools.postproc.correction.params import cluster_params as params
from argparse import ArgumentParser
from .opts import base_opts
import json
from doctools.cluster.mst import recluster
from collections import Counter, defaultdict
from copy import deepcopy

def sorcery(book):
    data, changed = pc.load(book, feat='images', **params)
    mdata, changed = pc.load(book, feat='ocr')
    edges = data["edges"]
    components = data["components"]
    n = data["vertices"]
    eset = set(mdata["errored"])

    edges, components = recluster(edges, n, threshold=0.1, rep='components')

    components = sorted(components, key=len)
    es = sorted(components, key=lambda x: len(eset.intersection(set(x))))

    def generate(candidate, ds):
        c = Counter()
        tc = Counter(ds[candidate])
        for t in ds:
            if t != candidate:
                c += Counter(ds[t])
        return c, tc

    def print_c(c, tc, candidate):
        #print("Correct,", candidate,'-', sum(tc.values()))
        def csvf(*args):
            _csvf = lambda ls: ','.join(map(str, ls))
            print(_csvf(args))

        csvf(candidate, tc[candidate], "proposal")
        for pred in tc:
            if pred != candidate:
                csvf(pred, tc[pred], "match")
        csvf(candidate, c[candidate], "rwe")
        #print("Incorrect,", sum(c.values()))
        for pred in c:
            if pred != candidate:
                csvf(pred, c[pred], "mismatch")

        print('-'*10)

    ls = []

    for component in components:
        cset = set(component)
        errored = eset.intersection(cset)
        errored = cset
        correct = cset.difference(eset)
        if errored:
            epreds = [mdata["predictions"][i] for i in errored]
            etruths = [mdata["truths"][i] for i in errored]
            ds = defaultdict(list)
            for t, p in zip(etruths, epreds):
                ds[t].append(p)
            xs = [mdata["predictions"][i] for i in component]
            candidate, count = max(Counter(xs).items(), key=lambda x: x[1])
            c, tc = generate(candidate, ds)
            ls.append((c, tc, candidate))

    def f(x):
        c, tc, candidate = x
        tc2 = deepcopy(tc)
        tc2[candidate] = 0
        return sum(tc2.values())

    ls = sorted(ls, key=f)
    for c, tc, candidate in ls:
        print_c(c, tc, candidate)

    

if __name__ == '__main__':
 
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()
    config_file = open(args.config)
    config = json.load(config_file)

    # Load OCR
    print(config["model"])
    #ocr = GravesOCR(config["model"], config["lookup"])
    book_name = config["books"][args.book]
    print("Book:", book_name)
    sorcery(book_name)

