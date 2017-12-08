import sys
import os 
import pdb
from doctools.cluster.mst import cluster
from doctools.cluster.distance import jaccard, lev, inv_jaccard
from argparse import ArgumentParser
from doctools.scripts.plot_tsne import tsne
import numpy as np
import pandas as pd
from functools import partial

def normal_lev(x, y):
	return lev(x, y, normalized=True)

parser = ArgumentParser()
parser.add_argument('-f', '--file', required=True)

args = parser.parse_args()

words = open(args.file).read().splitlines()[:1000]

M = cluster(words, normal_lev, threshold=8)


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