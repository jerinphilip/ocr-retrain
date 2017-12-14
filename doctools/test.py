from doctools.cluster.mst import cluster, merge
from doctools.cluster.distance import jaccard, lev, euc, cos
from doctools.parser.convert import page_to_unit
import pdb


def trial():
	
	with open('doctools/predictions.txt', 'r') as fp:
		lines = fp.readlines()
	predictions =[]
	for line in lines:
		line = line.split()
		try:
			predictions.append(line[1])
		except IndexError:
			print(line)
	edges, components = cluster(predictions, lev, 
            threshold=0.8, prune_above=0.8, rep='components')
	print(edges)

trial()