import doctools.parser.cluster as pc
from doctools.postproc.correction.params import cluster_params as params
from argparse import ArgumentParser
from .opts import base_opts
import json
from doctools.cluster.mst import recluster
from collections import Counter, defaultdict

def sorcery(book):
    data, changed = pc.load(book, feat='combined', **params)
    mdata, changed = pc.load(book, feat='ocr')
    edges = data["edges"]
    components = data["components"]
    n = data["vertices"]
    eset = set(mdata["errored"])

    edges, components = recluster(edges, n, threshold=0.1, rep='components')

    components = sorted(components, key=len)

    for component in components:
        cset = set(component)
        errored = eset.intersection(cset)
        #errored = cset
        correct = cset.difference(eset)
        if errored:
            epreds = [mdata["predictions"][i] for i in errored]
            etruths = [mdata["truths"][i] for i in errored]
            ds = defaultdict(list)
            for t, p in zip(etruths, epreds):
                ds[t].append(p)
            xs = [mdata["predictions"][i] for i in component]
            candidate, count = max(Counter(xs).items(), key=lambda x: x[1])
            cls = Counter(etruths)
            ls = sorted(cls.items(), key=lambda x: x[1])
            for c, f in ls:
                print(c, f)
                ps = Counter(ds[c])
                for p, f in ps.items():
                    print('\t', p, f)

            x = cls[candidate]
            y = sum(cls.values())
            print(candidate,"{}/{}".format(x, y))
            print('-'*10)
        else:
            #print("All correct")
            pass
            


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

