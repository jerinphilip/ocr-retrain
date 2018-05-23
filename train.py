import sys
import os
import torch
import pdb
import json
import pickle

from ocr.pytorch.util import *
from ocr.pytorch.engine import Engine
from ocr.pytorch.coding import Decoder
from parser.lookup import codebook

def squashed(pages):
	super_imgs, super_truths = [], []
	for imgs, truths, _, _ in pages:
		super_imgs.extend(imgs)
		super_truths.extend(truths)
	return (super_imgs, super_truths)

if __name__ == '__main__':
	all_pages = load(book='all_11', feat='pages')
	sequences, targets = squashed(all_pages[:100])
	samples = [(sequences[i], targets[i]) for i in \
							range(len(sequences)) if len(targets[i])!=0]

	lookup_filename = "lookups/Sanskrit.txt"
	lmap, ilmap = codebook(lookup_filename)
	convert_gpu = lambda x: list(map(gpu_format(lmap), x))
	# pdb.set_trace()
	# train, test = split(samples, split = 0.8, random = True)
	# train_set = convert_gpu(train)
	# test_set = convert_gpu(test)
	train_set = convert_gpu(samples)
	# print(train_set[1])
	satisfactory = False
	savepath = "file.tar"
	checkpoint = {}
	arch = {
			"input_size": 32,
			"hidden_size": 50,
			"output_classes": len(lmap.keys()),
            "lr": 3e-4,
            "momentum": 0.8,
	}
	try:
	    with open(savepath, "rb") as savefile:
	    	checkpoint = torch.load(savepath)
	    	start_epoch = checkpoint['epoch']
	    	print("=> loaded checkpoint '{}' (trained for {} epochs)".format(savepath, checkpoint['epoch']))
	except FileNotFoundError:
		print('file not found')
		checkpoint = {
				
	            "epoch":0,
	            "best":float("inf"), 

	    }

	engine = Engine(savepath=savepath, **checkpoint, **arch)
	print('training..')
	val_err, train_err = engine.train(train_set, debug=True)
	# engine.test(test_set)
	# decoder  = Decoder(lmap, ilmap)

	# for pair in (test_set):
	#     sequence, target = pair
	#     prediction = engine.recognize(sequence, decoder)
	#     print('Prediction: %s'%prediction)
	
	# kwargs = engine.export()
	# torch.save(kwargs, open(savepath, "wb+"))
	