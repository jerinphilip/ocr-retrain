from .params import params
from .base import naive, suggest, cluster

from doctools.scripts.opts import base_opts
from argparse import ArgumentParser
from doctools.parser.convert import page_to_unit
from doctools.parser import read_book, text
from doctools.postproc.dictionary import Dictionary
from doctools.ocr import GravesOCR
from doctools.parser.nlp import extract_words
from doctools.meta.file_locs import get_pickeled, get_clusters
import json
import pickle
import os 
import pdb
from doctools.scripts.debug import time

@time
def loov(book_locs, book_index, em):
	full_text = '\n'.join(list(map(text, book_locs)))
	print('Adding words to Vocabulary...')
	
	words = extract_words(full_text)
	em.enhance_vocab_with_books(words)
	
	return em

@time
def classify_errors(predictions, truths, em):
	
	print("classifying errors...")
	correctable, uncorrectable, correct, real_world_errors = 0, 0, 0, 0
	errors = [em.error(prediction) for prediction in predictions]
	error_indices = [i for i,v in  enumerate(errors) if v != 0]
	correct_indices = [i for i,v in  enumerate(errors) if v == 0]
	
	for index in correct_indices:
		if predictions[index] == truths[index]:
			correct+=1
		else:
			real_world_errors+=1
	
	print("this will take time ..... :(")
	for index in error_indices:
		suggestions = em.suggest(predictions[index])
		if truths[index] in suggestions:
			correctable += 1
		else:
			uncorrectable += 1
			em.enhance_vocabulary(truths[index])
	print("oh!!! it's done... :D")
	return correct, real_world_errors, correctable, uncorrectable
if __name__ == '__main__':
	parser = ArgumentParser()
	base_opts(parser)
	args = parser.parse_args()
	config_file = open(args.config)
	config = json.load(config_file)
	print(config["model"])
	ocr = GravesOCR(config["model"], config["lookup"])
	error_module = Dictionary(**config["error"])
	outpath = args.output
	# defining paths for books and predictions
	book_index = args.book
	book_list = params["books"]
	for book_index in range(len(book_list)):
		book_name = book_list[book_index]
		# pdb.set_trace()
		book_locs = list(map(lambda x: config["dir"] + x + '/', book_list))
		
		new_error_module = loov(book_locs, book_index, error_module)

		fpath = os.path.join(config["dir"], book_name)
		print(book_name)
		print("Reading book...", end='', flush=True)
		pagewise = read_book(book_path=fpath, unit='word')
		num_pages = int(len(pagewise))
		print("Done")
		images, truths = page_to_unit(pagewise)
		# Load the predictions
		predictions = get_pickeled(book_name, type="predictions")
		correct, real_world_errors, correctable, uncorrectable = classify_errors(predictions, truths, new_error_module)
		# load clusters
		
		
		# pdb.set_trace()
		print("Calculating cost for Naive...")
		cost_naive, error_naive = naive(predictions, truths, error_module)

		print("Calculating cost for Suggest...")
		cost_suggest, error_suggest = suggest(predictions, truths, new_error_module)
		
		print("Calculating cost for image clusters")
		data = get_clusters(book_name, features="images")
		cost_cluster_images, error_cluster_images = cluster(predictions, truths, error_module, data)

		print("Calculating cost for word clusters")
		data = get_clusters(book_name, features="words")
		cost_cluster_words, error_cluster_words = cluster(predictions, truths, error_module, data)
		print("Calculating cost for combined clusters")
		data = get_clusters(book_name, features="combined")
		cost_cluster_combined, error_cluster_combined = cluster(predictions, truths, error_module, data)
		
		print("Saving stats..")
		cost_dict = { 	"Pages": num_pages,

						"word stats":{
						"correct": correct,
						"real world errors": real_world_errors,
						"correctable": correctable,
						"uncorrectable": uncorrectable 
						},
					"cost ":{
						"naive": cost_naive,
						"suggest":cost_suggest,
						"cluster":{
							"words":cost_cluster_words,
							"images":cost_cluster_images,
							"combined":cost_cluster_combined
							}
						},
					"errors":{
						"naive":error_naive,
						"suggest":error_suggest,
						"cluster":{
							"words":error_cluster_words,
							"images":error_cluster_images,
							"combined":error_cluster_combined
								}
						}
					
		}
		with open('%s/jsons_cost/%s.json'%(outpath, book_name), 'w+') as fp:
			json.dump(cost_dict, fp, indent=4)
		print("finished..")
		# print(cost_naive, error_naive)
		# print (cost, error)
		# print(cost_cluster, error_cluster)

			






  #   if os.path.exists(os.path.join('%s'%outpath_json,'%s.json'%book_name)):
		# print('Loading clusters ...')
		# with open(os.path.join('%s'%outpath_json,'%s.json'%book_name), 'r') as json_data:
		# 	data = json.load(json_data)
		# print('Done....')