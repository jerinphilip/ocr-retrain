from .params import params
from .base import naive, suggest
from doctools.scripts.opts import base_opts
from argparse import ArgumentParser
from doctools.parser.convert import page_to_unit
from doctools.parser import webtotrain
from doctools.postproc.dictionary import Dictionary
from doctools.ocr import GravesOCR
from doctools.parser.nlp import extract_words
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
	error = Dictionary(**config["error"])

	# defining paths for books and predictions
	book_index = args.book
	book_list = params["books"]
	book_name = book_list[book_index]
	book_locs = list(map(lambda x: config["dir"] + x + '/', book_list))
	# enhancing Vocabulary
	error = loov(book_locs, book_index, error)
	fpath = os.path.join(config["dir"], book_name)
	outpath_pickled = os.path.join(args.output, 'pickled')
	print("Reading book...", end='', flush=True)
	pagewise = webtotrain.read_book(fpath)
	print("Done")
	images, truths = page_to_unit(pagewise)
	# Load the predictions
	predictions = None
	if os.path.exists(os.path.join('%s'%outpath_pickled,'%s.pkl'%book_name)):
		print('Loading predictions...')
		with open(os.path.join('%s'%outpath_pickled,'%s.pkl'%book_name), 'rb') as f:
			predictions = pickle.load(f)
		print('Done....')

	cost, error = suggest(predictions, truths, error)
	cost_naive, error_naive = naive(predictions, truths, error)
	print(cost_naive)
	print (cost)

    