from .params import params
from .base import naive, suggest, cluster
from doctools.cluster.mst import cluster, merge
from doctools.scripts.opts import base_opts
from argparse import ArgumentParser
from doctools.parser.convert import page_to_unit
from doctools.parser import webtotrain
from doctools.postproc.dictionary import Dictionary
from doctools.ocr import GravesOCR
from doctools.parser.nlp import extract_words
from doctools.meta.file_locs import get_pickeled, get_clusters
import json
import pickle
import os 
import pdb
from time import time

def loov(book_locs, book_index, em):
	full_text = '\n'.join(list(map(webtotrain.full_text, book_locs)))
	print('Adding words to Vocabulary...')
	t0 = time()
	words = extract_words(full_text)
	em.enhance_vocab_with_books(words)
	print('done in %.2fs.' % (time() - t0))
	return em
if __name__ == '__main__':
	parser = ArgumentParser()
	base_opts(parser)
	args = parser.parse_args()
	config_file = open(args.config)
	config = json.load(config_file)
	print(config["model"])
	ocr = GravesOCR(config["model"], config["lookup"])
	error_module = Dictionary(**config["error"])

	# defining paths for books and predictions
	book_index = args.book
	book_list = params["books"]
	book_name = book_list[book_index]
	book_locs = list(map(lambda x: config["dir"] + x + '/', book_list))
	# enhancing Vocabulary
	# pdb.set_trace()
	new_error_module = loov(book_locs, book_index, error_module)

	fpath = os.path.join(config["dir"], book_name)
	outpath_pickled = os.path.join(args.output, 'pickled')
	outpath_json = os.path.join(args.output, 'jsons')
	print("Reading book...", end='', flush=True)
	pagewise = webtotrain.read_book(fpath)
	print("Done")
	images, truths = page_to_unit(pagewise)
	# Load the predictions
	predictions = get_pickeled(book_name, type="predictions")
	
	# pdb.set_trace()
	data = get_clusters(book_name, features="words")
	
	cost, error = suggest(predictions, truths, new_error_module)
	cost_naive, error_naive = naive(predictions, truths, error_module)
	cost_cluster, error_cluster = cluster(predictions, truths, error_module, data)
	pdb.set_trace()
	print(cost_naive)
	print (cost)
	print(cost_cluster)
    






  #   if os.path.exists(os.path.join('%s'%outpath_json,'%s.json'%book_name)):
		# print('Loading clusters ...')
		# with open(os.path.join('%s'%outpath_json,'%s.json'%book_name), 'r') as json_data:
		# 	data = json.load(json_data)
		# print('Done....')