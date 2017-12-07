from doctools.parser.hwnet import read_annotation
from doctools.meta.file_locs import get_pickeled
import os
from shutil import copy
from collections import defaultdict
from pprint import pprint
import numpy as np
from doctools.postproc.correction.params import params

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

    graph = get_pickeled(book_id, type='edges')
    group_path = os.path.join(base_path, "group", book_id)
    guarded_mkdir(group_path)

    print(len(graph["components"]))
    input()

    for i, component in enumerate(graph["components"]):
        # Create directory
        component_path = os.path.join(group_path, str(i))
        guarded_mkdir(component_path)
        for j in component:
            src = os.path.join(book_path, imgs[j])
            guarded_copy(src, component_path)
            #print(imgs[j], truths[j], j)
            print(truths[j])
        input()

def find_distance(book_id):
    base_path = '/OCRData2/praveen-intermediate/'
    book_path = os.path.join(base_path, book_id)
    fpath = os.path.join(book_path, 'annotation.txt')
    base, imgs, truths = read_annotation(fpath)

    graph = get_pickeled(book_id, type='edges')
    print(book_id)

    def create_adj(edges, threshold):
        adj = defaultdict(list)
        for link in edges:
            u, v = link
            w = edges[link]
            if w < threshold:
                adj[u].append(v)
                adj[v].append(u)
        return adj

    #adj = create_adj(graph["edges"], 0.2)

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
    book_id = '0005'
    for book_id in params["books"]:
        try:
            find_distance(book_id)
        except:
            pass
