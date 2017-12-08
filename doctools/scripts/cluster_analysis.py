import json
from argparse import ArgumentParser
import pdb
from doctools.ocr import GravesOCR
import os
from .opts import opts_v02, base_opts
from doctools.parser import webtotrain
from doctools.parser.convert import page_to_unit
from .debug import time
import pandas as pd 
from doctools.cluster.distance import jaccard, lev, euc, cos
import numpy as np
from doctools.postproc.dictionary import Dictionary
from doctools.meta.file_locs import get_pickeled, get_clusters
import pickle
import numpy as np
@time
def get_units(fpath):
    print("Reading book...", end='', flush=True)
    pagewise = webtotrain.read_book(fpath)
    pagewise = pagewise
    print("Done")
    images, truths = page_to_unit(pagewise)
    return images, truths
def compare(key, values, predictions, em):
	
	try:
		errors = sum([em.error(predictions[int(values[i])]) for i in range(len(values))])
	
		correct = len(values) - errors
	except IndexError:
		print("error... closing")

		
	return errors, correct

def foo(E, features, predictions, book):
	row = []
	for link, weight in E.items():
		u1, v1 = link
		dist = cos(features[u1], features[v1])
		row.append([predictions[u1], predictions[v1], dist])
	df = pd.DataFrame(row, columns=['U', 'V', 'Dist'])
	df = df.sort_values(by='Dist')
	df.to_csv('results/%s_thresh.csv'%book)
	print('Thresh computation Done.....')

if __name__ == '__main__':
	parser = ArgumentParser()
	base_opts(parser)
	args = parser.parse_args()
	config_file = open(args.config)
	config = json.load(config_file)	
	# book = args.book
	book_list = ['0022','0029','0060','0061','0069', '0191', '0211']
	
	book_name = book_list[args.book]

	print("Reading book...", end='', flush=True)
	pagewise = webtotrain.read_book(os.path.join(config["dir"], book_name))
	print("Done")

	print('Loading features...')
	feat = np.load(os.path.join(config["feat_dir"], '%s'%book_name, "feats.npy"))
	features = [feat[i] for i in range(feat.shape[0])]
	print('Done....')
    
	images, truths = page_to_unit(pagewise)
	path = os.path.join(args.output, 'jsons_feat')
	outpath_pickled = os.path.join(args.output, 'pickled/')
	
	
	ocr = GravesOCR(config["model"], config["lookup"])
	error = Dictionary(**config["error"])
	
	edges = get_pickeled(book_name, type="edges")
	

	for book in books:
		print('%s book under process'%book)
		print(config["model"])
		
		fpath = os.path.join(config["dir"], book)
		images, truths = get_units(fpath)
		predictions = get_pickeled(book_name, type="predictions")
		data = get_clusters(book_name, features="words")
		val, key, row  = [],[], []
		# pdb.set_trace()
		for k,v in data.items():
			
			err, corr = compare(k, v, predictions, error)
				# pdb.set_trace()
				
			row.append([k,len(v), corr,  err])
		df = pd.DataFrame(row,  columns=['Exemplar','All', 'Correct','Error'])
		mean_all = np.ceil(np.mean(df['All']))
		
		mean_correct = np.ceil(np.mean(df['Correct']))
		mean_error = np.ceil(np.mean(df['Error']))
		df = df.sort_values(by='All',  ascending=[0])
		# pdb.set_trace()
		df.to_csv('results/%s_feat_stats.csv'%book)
		print('Done.....')

