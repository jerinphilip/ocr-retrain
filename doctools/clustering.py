import sys
import os 
src = os.path.abspath("../ocr/")
sys.path.insert(0, '../src/doctools/')
from doctools.cluster.mst import cluster
from doctools.cluster.distance import jaccard, lev
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-f', '--file', required=True)

args = parser.parse_args()

words = open(args.file).read().splitlines()
M = cluster(words, lev, threshold=8)
# print(M)
m, n = M.shape
for i in range(m):
    for j in range(n):
        if M[i][j]:
            print(i, j, M[i][j])

