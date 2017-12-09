from doctools.parser.hwnet import read_annotation
from doctools.meta.file_locs import get_pickeled
import os
from shutil import copy
from collections import defaultdict
from pprint import pprint
import numpy as np
from doctools.postproc.correction.params import params
from doctools.cluster.mst import recluster
import doctools.parser.cluster as pc
from doctools.postproc.correction.params import cluster_params as cparams

def guarded_copy(src, dest):
    #print("rsync", src, dest)
    copy(src, dest)

def guarded_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def group(book_id):
    base_path = '/OCRData2/praveen-intermediate/'
    book_path = os.path.join(base_path, book_id)
    fpath = os.path.join(book_path, 'annotation.txt')
    base, imgs, truths = read_annotation(fpath)


    for feat in ["images", "words"]:
        print(cparams[feat])
        graph, changed = pc.load(book_id, feat=feat, **cparams[feat])
        group_path = os.path.join(base_path, "group", book_id, feat)
        guarded_mkdir(group_path)

        components = sorted(graph["components"], key=len, reverse=True)

        for i, component in enumerate(components):
            print(len(component))
            # Create directory
            component_path = os.path.join(group_path, str(i))
            guarded_mkdir(component_path)
            for j in component:
                try:
                    src = os.path.join(book_path, imgs[j])
                    guarded_copy(src, component_path)
                    #print(imgs[j], truths[j], j)
                    print(truths[j])
                except IndexError:
                    pass
            input()

def find_distance(book_id):
    base_path = '/OCRData2/praveen-intermediate/'
    book_path = os.path.join(base_path, book_id)
    fpath = os.path.join(book_path, 'annotation.txt')
    base, imgs, truths = read_annotation(fpath)

    graph = get_pickeled(book_id, type='edges')

    # n = sum([len(c) for c in graph["components"]])
    n = 14686
    print(len(truths), n)
    input()
    edges, components = recluster(graph["edges"], n, rep='components', threshold=0.2)


    occurrences = defaultdict(list)

    for i, truth in enumerate(truths):
        occurrences[truth].append(i)

    weights = defaultdict(list)
    means = []
    for truth in occurrences:
        ls = occurrences[truth]
        n = len(ls)
        for i in range(n) :
            for j in range(i+1, n):
                u, v = ls[i], ls[j]
                if (u, v) in graph["edges"]:
                    w = graph["edges"][(u, v)]
                    weights[truth].append(w)
        n = len(weights[truth])
        if n:
            v = sum(weights[truth])/n
            means.append(v)

    print(np.mean(means), np.std(means))

if __name__ == '__main__':
    book_id = '0037'
    #for book_id in params["books"]:
    #find_distance(book_id)
    group(book_id)
