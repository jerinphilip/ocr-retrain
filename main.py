import sys
import os
import torch
import pdb
import json
from parser.loader import read_book
import torch.utils.data as tud
from torchvision import transforms, utils
from argparse import ArgumentParser
from parser.opts import base_opts
from ocr.preproc import DataLoader
from ocr.pytorch.util import time, load, save, split
from parser.lookup import codebook
from tqdm import *
import numpy as np
from ocr.pytorch.engine import Engine

class OCRDataset(tud.Dataset):
	def __init__(self, **kwargs):
		self.config = kwargs['config']
		self.dir = self.config['dir']
		self.books = self.config["books"]
		self.transform = kwargs['transform']
	def __len__(self):
		return len(self.books)
	def __getitem__(self, idx):
		samples = load(book=self.books[idx], feat='feat')
		if samples is None:
			pagewise = read_book(book_path=os.path.join(self.dir, self.books[idx]))
			print('len pagewise:%d'%len(pagewise))
			loader = DataLoader(pagewise=pagewise[2:])
			sequences, targets = loader.sequences, loader.targets
			samples = [(sequences[i], targets[i]) for i in range(len(sequences)) if len(targets[i])!=0]
			if self.transform:
				samples = self.transform(samples)
			save(book=self.books[idx], data=samples, feat='feat')
		return samples
		
class ToTensor(object):
	def __init__(self, lmap):
		self.lmap = lmap
	def gpu_format(self, sample):
		seq, targ = sample
		seq = torch.Tensor(seq.astype('float32'))
		seq = seq.unsqueeze(0)
		seq = seq.permute(2, 0, 1).contiguous()
		targ = [lmap[x] if x in lmap else lmap['!'] for x in targ]
		targ = torch.IntTensor(targ)
		return (seq, targ)
	def __call__(self, samples):
		f = lambda x: list(map(self.gpu_format, x))
		return f(samples)
def train_test(train_set, test_set):
	savepath = "file.tar"
	kwargs = {}
	try:
	    with open(savepath, "rb") as savefile:
	        kwargs = torch.load(savefile)
	except FileNotFoundError:
	    kwargs = {
	            "input_size": 32,
	            "output_classes": len(lmap.keys()),
	            "lr": 3e-4,
	            "momentum": 0.8
	    }
	engine = Engine(**kwargs)
	print('Training')
	val_err, train_err = engine.train(train_set, debug=True)
	engine.test(test_set)
	kwargs = engine.export()
	torch.save(kwargs, open(savepath, "wb+"))

if __name__ == '__main__':
	parser = ArgumentParser()
	base_opts(parser)
	args = parser.parse_args()
	config_file = open(args.config)
	config = json.load(config_file)
	lang = args.lang
	lookup = args.lookup
	lmap, ilmap = codebook(args.lookup)
	transformed_dataset = OCRDataset(config=config, transform=transforms.Compose([ToTensor(lmap)]))
	total = torch.zeros
	total = []
	for i in trange(len(transformed_dataset)):
		sample = transformed_dataset[i]
		total+=(sample)
	train, test = split(total, random=True, split=0.8)
	train_test(train, test)

