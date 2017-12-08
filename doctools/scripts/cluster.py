from doctools.cluster.mst import cluster, merge
from doctools.cluster.distance import jaccard, lev, euc, cos
from doctools.parser.convert import page_to_unit
from doctools.meta.file_locs import get_pickeled, get_clusters
from doctools.cluster.k_nearest.distance import normalized_euclid_norm
from doctools.parser import read_book
from argparse import ArgumentParser
from pprint import pprint
import json
from functools import partial
import pdb
from doctools.ocr import GravesOCR
import os
from .opts import base_opts
import numpy as np
import cv2
import pdb
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from .debug import time
import pickle
import doctools.parser.cluster as pc

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


@time
def get_predictions(fpath):
    print("Reading book...", end='', flush=True)
    pagewise = read_book(book_path=fpath, unit='word')
    
    print("Done")
    images, truths = page_to_unit(pagewise)
    print("Predicting....", end='', flush=True)
    predictions = ocr.predict(images)
    print("Done")   
    errored = [predictions[i] for i in range(len(truths)) if predictions[i] != truths[i]]
    return errored, predictions

def get_images(fpath):
    print("Reading book...", end='', flush=True)
    pagewise = read_book(book_path=fpath, unit='word')
    print("Done")
    images, truths = page_to_unit(pagewise)
    return images, truths

@time
def get_features(fpath):
    print('Loading features...')
    feat = np.load(os.path.join(fpath, "feats.npy"))
    features = [feat[i] for i in range(feat.shape[0])]
    print('Done....')
    return features

@time
def form_clusters(elements, **kwargs):
    dist  = kwargs["distance"]
    threshold = kwargs["threshold"]
    print("Clustering....", end='', flush=True)
    edges, components = cluster(elements, dist, threshold=threshold,prune_above=0.8, rep='components')
    print("Done")
    return edges, components


if __name__ == '__main__':
 
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()
    config_file = open(args.config)
    config = json.load(config_file)

    # Load OCR
    print(config["model"])
    ocr = GravesOCR(config["model"], config["lookup"])
    book_name = config["books"][args.book]
    print("Book:", book_name)

    outpath = args.output
    outpath_pickled = os.path.join(args.output, 'pickled')

    # Load features and images

    def feats(book_name):
        fpath = os.path.join(config["feat_dir"], book_name)
        return get_features(fpath)

    def predict(book_name):
        fpath = os.path.join(config["dir"], book_name)
        images, truths = get_images(fpath)
        return ocr.predict(images)

    params = {
        "words": {
            "distance" : lev,
            "threshold" : 0.5
        },
        "images": {
            "distance": normalized_euclid_norm,
            "threshold" : 0.36
        }
    }


    loader = {
        "images":  feats,
        "words":  predict
            
    }

    data = {}
    for feat in params:
        X, changed = pc.load(book_name, **params[feat], feat=feat)
        if X is None or changed:
            X = loader[feat](book_name)
            edges, components = form_clusters(X, feat=feat, **params[feat])
            data[feat] = {"components": components, "edges": edges, "vertices": len(X)}
            pc.save(data = data[feat], book=book_name, feat=feat, params=params[feat], outpath=outpath_pickled)
        else:
            data[feat] = X

    # Finally do the thing for combined.
    X, changed = pc.load(book_name, **params, feat='combined')
    if X is None or changed:
        edges, components = merge(data["words"]["edges"], 
                data["images"]["edges"], 
                data["images"]["vertices"])
        data["combined"] = {"components": components, "edges": edges, "vertices": data["images"]["vertices"]}
        pc.save(data = data["combined"], book=book_name, feat="combined", params=params, outpath=outpath_pickled)
