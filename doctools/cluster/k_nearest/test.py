from .k import k_neighbours
from .distance import euclid_norm, normalized_euclid_norm
from .checks import checks
from doctools.scripts.opts import base_opts
from doctools.postproc.correction.params import params
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
import numpy as np

def get_features(fpath):
    print('Loading features...')
    feat = np.load(os.path.join(fpath, "feats.npy"))
    features = [feat[i] for i in range(feat.shape[0])]
    print('Done....')
    return features

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


	fpath = os.path.join(config["dir"], book_name)
	outpath_pickled = os.path.join(args.output, 'pickled')
	outpath_json = os.path.join(args.output, 'jsons')
	# print("Reading book...", end='', flush=True)
	# pagewise = webtotrain.read_book(fpath)
	# print("Done")
	# images, truths = page_to_unit(pagewise)
	# Load the predictions
	predictions = get_pickeled(book_name, type="predictions")
	features = get_features(os.path.join(config["feat_dir"], book_name))
	checks(predictions, features)


	print("Finished")
	# k, components = 3, []
	# for u in predictions:
	# 	vs = k_neighbours(u, predictions, k)
	# 	for V in vs:
	# 		checks(V)
	# 	components.append(vs)

	# 