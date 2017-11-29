import json
from argparse import ArgumentParser
import pdb
from doctools.ocr import GravesOCR
import os
from .opts import opts_v02
from doctools.parser import webtotrain
from doctools.parser.convert import page_to_unit
from .debug import time

@time
def get_units(fpath):
    print("Reading book...", end='', flush=True)
    pagewise = webtotrain.read_book(fpath)
    print("Done")
    images, truths = page_to_unit(pagewise)
    return images, truths

if __name__ == '__main__':
	parser = ArgumentParser()
	opts_v02(parser)
	args = parser.parse_args()
	book = args.book
	path = args.outpath
	config_file = open(args.config)
	config = json.load(config_file)
	
	print(config["model"])
	ocr = GravesOCR(config["model"], config["lookup"])
	fpath = os.path.join(config["dir"], book)
	images, truths = get_units(fpath)
   
	with open('%s/%s.json'%(path,book)) as json_data:
		data = json.load(json_data)
	count = 0
	for k,v in data.items():
		count+= len(data[k])
	print("total words in cluster: %d \n total ground truths: %d "%(count, len(truths)))
	pdb.set_trace()