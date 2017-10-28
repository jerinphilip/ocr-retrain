from doctools.cluster.mst import cluster
from doctools.cluster.distance import jaccard, lev
from argparse import ArgumentParser
from pprint import pprint
from .dot import as_dot
from distance import levenshtein
import json
from functools import partial

lev2 = partial(levenshtein, normalized=True)

parser = ArgumentParser()
parser.add_argument('-f', '--file', required=True)
parser.add_argument('-o', '--output', required=True)
args = parser.parse_args()

words = open(args.file).read().splitlines()
edges, components = cluster(words, lev2, threshold=0.5, prune_above=0.8, rep='components')

for c, component in enumerate(components):
    rep = as_dot(words, edges, component) 
    of = "{}_{}.dot".format(args.output, c)
    with open(of, "w+") as ofp:
        ofp.write(rep)
