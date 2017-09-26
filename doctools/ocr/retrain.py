import os
from ocr import GravesOCR
import numpy as np
import parser.webtotrain as webtotrain
from error_module.dictionary import Dictionary
from Levenshtein import distance
import json
import sys
import re
import pandas as pd
def split_index(data, num_first):
    #assert(0 <= fraction and fraction <= 1)
    total = len(data)
    split_index = 0
    for i in range(num_first):
        split_index += len(data[i][0])
    print("Split Index:", split_index)
    return split_index



           
def ocr_accuracy(ocr,em,book_path,lookup,book_index):
	row=[]
	epochs=2 
	pagewise= webtotrain.read_book(book_path)
	#pagewise=pagewise[:10]
	num_pages = len(pagewise)
	images,truths=[],[]
	for im,tr in pagewise:
		images.extend(im)																											
		truths.extend(tr)
	batch_size=10
	
	for npage_to_include in range(0, num_pages, batch_size):
		if npage_to_include == 0:
			pass
		else:
			print("pages to include: %d"%npage_to_include)
			si = split_index(pagewise, npage_to_include)
			print("Number of sequences: "+str(si))
			images_chunk = images[:si]
			truths_chunk = truths[:si]
			total_chars = ocr.no_of_characters(truths_chunk)
			print("Computing Errors, with %d pages included."%(npage_to_include), flush=True)
			predictions = [ocr.recognize(image) for image in images_chunk]
			errors = [em.error(prediction) for prediction in predictions]
			threshold = 0
			edit_dist=0
			indices = [i for i, x in enumerate(errors) if x == 1]
			#print(indices, len(indices))
			#print("truth value: "+truths_chunk[9]+"\n","prediction: "+predictions[9]+"\n")
			for i in range(len(truths_chunk)):
			  	truth, prediction, error = truths_chunk[i], predictions[i], errors[i]
			  	

			  	if error > threshold:
			  		edit_dist+=distance(truth, prediction) 

			ler = (edit_dist/total_chars)*100
			row.append([npage_to_include, si, ler, edit_dist, total_chars])
	return (row)	 

if __name__ == '__main__':
	config = json.load(open(sys.argv[1]))
	book_index = int(sys.argv[2])
	lang = sys.argv[3]
	output_dir = 'new_ocr_stats'
	ocr = GravesOCR(config["model"], config["lookup"])
	book_locs = list(map(lambda x: config["dir"] + x + '/', config["books"]))
	error = Dictionary(**config["error"])
	row = ocr_accuracy(ocr,error,book_locs[book_index],config["lookup"],book_index)
	df = pd.DataFrame(row, columns=["Pages", "Sequences", "LabeErrorRate", "Editdistance", "TotalChar"])
	df.to_csv('%s/%s/stats_%s.csv'%(output_dir, lang, config["books"][book_index]))
	print("done....")
	