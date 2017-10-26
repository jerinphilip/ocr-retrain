import sys
import os 
import pdb
src = os.path.abspath("../ocr/")
sys.path.insert(0, '../src/doctools/')
from doctools.cluster.mst import cluster
from doctools.cluster.distance import jaccard, lev
from argparse import ArgumentParser
from doctools.scripts.plot_tsne import tsne
import numpy as np
import pandas as pd
parser = ArgumentParser()
parser.add_argument('-f', '--file', required=True)

args = parser.parse_args()

words = open(args.file).read().splitlines()[:500]
M = cluster(words, lev, threshold=8)


# print(M)
m, n = M.shape

tsne(M)
# 
# for i in range(m):
#     for j in range(n):
#         if M[i][j]:
#             print(i, j, M[i][j])

row, col = np.nonzero(M)
row, col = list(row), list(col)
tup = list(zip(row,col))
exemplars = np.unique(row)
mydict={}
for ex in exemplars:
	mydict[words[ex]] =[]
for i,j in tup:
	
	mydict[words[i]].append(words[j])

df = pd.DataFrame(list(mydict.items()), columns=['exemplars', 'clusters'])
df.to_csv('mst.csv')