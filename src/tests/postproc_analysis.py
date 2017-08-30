import sys
import os
src = os.path.abspath("../ocr/")
sys.path.insert(0, '../src/')
sys.path.insert(0, '../src/meta/')
from ocr import GravesOCR
from postproc.dictionary import Dictionary
from parser import read_book
import json
from cost_model import CostModel
from timekeep import Timer
from parser.convert import page_to_unit
import parser.webtotrain as webtotrain
from parser.nlp import extract_words
from collections import Counter
import pdb


def init_dict(mydict, truths):
	truths = set(truths)
	for truth in truths:
		mydict[truth] = list()
	return mydict
def count(error_list):
	count = Counter()
	d={}
	for item in error_list:
		count.update([item])
	for key, value in count.items():

		d[key] = value
	return d

def analyze(ocr, em, book_locs, book_index):
	timer = Timer(debug=True)
	book_path = book_locs.pop(book_index)
	full_text = '\n'.join(list(map(webtotrain.full_text, book_locs)))
	words = extract_words(full_text)
	em.enhance_vocab_with_books(words)
	timer.start("read images")
	pagewise = webtotrain.read_book(book_path)
	#pagewise = pagewise
	page_count = len(pagewise)
	images, truths = page_to_unit(pagewise)
	n_images = len(images)
	intervals = 20.0
	batchSize = int(n_images/intervals)
	timer.start("ocr, recognize")
	predictions = [ocr.recognize(image) for image in images]
	timer.end()
	threshold = 0
	wo_dict = {}
	gt_dict = {}
	errors = [em.error(prediction) for prediction in predictions]
	wo_dict = init_dict(wo_dict, truths)
	gt_dict = init_dict(gt_dict, truths)
	for i in range(len(truths)):
		truth, prediction, error = truths[i], predictions[i], errors[i]
		gt_dict[truth].append(1)
		if error > threshold:
			if prediction:
				#pdb.set_trace()
				
				wo_dict[truth].extend([prediction])
			else:
				print('%s:%s'%(truth,prediction))

	proc={}
	#pdb.set_trace()
	for key in wo_dict.keys():
		if wo_dict[key]:
			proc[key] = count(wo_dict[key])

	for key in gt_dict.keys():
		if gt_dict[key]:
			gt_dict[key] = sum(gt_dict[key])

	return proc, gt_dict





if __name__ == '__main__':
	config = json.load(open(sys.argv[1]))
	book_index = int(sys.argv[2])
	lang = sys.argv[3]
	output_dir = 'postproc_analysis'
	ocr = GravesOCR(config["model"], config["lookup"])
	error = Dictionary(**config["error"])
	book_locs = list(map(lambda x: config["dir"] + x + '/', config["books"]))
	err_count, gt_count = analyze(ocr, error, book_locs, book_index)
	with open('%s/%s/error_%s.json'%(output_dir, lang,  config["books"][book_index]), 'w+') as fp:
		json.dump(err_count, fp, indent=4)
	with open('%s/%s/gt_%s.json'%(output_dir, lang,  config["books"][book_index]), 'w+') as fp:
		json.dump(gt_count, fp, indent=4)
