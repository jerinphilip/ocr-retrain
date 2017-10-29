from doctools.cluster.mst import cluster
from doctools.cluster.distance import jaccard, lev
from doctools.parser.convert import page_to_unit
from doctools.parser import webtotrain
from argparse import ArgumentParser
from pprint import pprint
from .dot import as_dot
from distance import levenshtein
import json
from functools import partial
from doctools.ocr import GravesOCR
import os
from .opts import base_opts

if __name__ == '__main__':
    lev2 = partial(levenshtein, normalized=True)
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()
    config_file = open(args.config)
    config = json.load(config_file)

    # Load OCR
    ocr = GravesOCR(config["model"], config["lookup"])

    # Parse Book in and predict
    book_name = config["books"][args.book]
    fpath = os.path.join(config["dir"], config["books"][args.book])

    print("Reading book...", end='', flush=True)
    pagewise = webtotrain.read_book(fpath)
    #pagewise = pagewise[5:7]
    print("Done")

    images, truths = page_to_unit(pagewise)

    print("Predicting....", end='', flush=True)
    predictions = ocr.predict(images)
    print("Done")

    # Cluster

    errored = [predictions[i] for i in range(len(truths)) if predictions[i] != truths[i]]
    print("Clustering....", end='', flush=True)
    edges, components = cluster(errored, lev2, threshold=0.5, prune_above=0.8, rep='components')
    print("Done")

    for c, component in enumerate(components):
        #rep = as_dot(predictions, edges, component) 
        rep = as_dot(errored, edges, component) 
        of = os.path.join(args.output, "{}.dot".format(c))
        #of = "{}_{}.dot".format(args.output, c)
        with open(of, "w+") as ofp:
            ofp.write(rep)
