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
from ocr.pytorch.coding import Decoder
import pdb
class OCRDataset(tud.Dataset):
	def __init__(self, **kwargs):
		self.config = kwargs['config']
		self.dir = self.config['dir']
		self.books = self.config["books"]
		self.transform = kwargs['transform']
		self.vocab = {}
		print(self.books)
	def __len__(self):
		return len(self.books)

	def __getitem__(self, idx):
		samples = load(book=self.books[idx], feat='feat')
		# pdb.set_trace()
		if samples is None:
			print('No samples found')
			pagewise= read_book(book_path=os.path.join(self.dir, self.books[idx]))
			loader = DataLoader(pagewise=pagewise)
			sequences, targets = loader.sequences, loader.targets
			samples = [(sequences[i], targets[i]) for i in range(len(sequences)) if len(targets[i])!=0]

			if self.transform:
				samples = self.transform(samples)
			save(book=self.books[idx], data=samples, feat='feat')
		return samples
		
class ToTensor(object):
	def __init__(self):
		self.lmap = load(book='Sanskrit', feat='gpu_lookup')
		if self.lmap is None:
			self.lmap = {}
	def update_lookup(self,lmap, key):
		lmap[key] = len(lmap.keys())
		save(book='Sanskrit', data=lmap, feat='gpu_lookup')
		return lmap
	def gpu_format(self, sample):
		seq, targ = sample
		seq = torch.Tensor(seq.astype('float32'))
		seq = seq.unsqueeze(0)
		seq = seq.permute(2, 0, 1).contiguous()
		for t in targ:
			if t not in self.lmap:
				print("Updating Dictionary with %s"%t)
				self.lmap = self.update_lookup(self.lmap, t)
		targ = [self.lmap[x] for x in targ]
		# pdb.set_trace()
		targ = torch.IntTensor(targ)
		if torch.cuda.is_available():
			seq = seq.cuda()
			# targ = seq.cuda()
		return (seq, targ)
	def __call__(self, samples):
		f = lambda x: list(map(self.gpu_format, x))
		return f(samples)
def train_test(train_set, test_set):
	savepath = "file_1.tar"
	kwargs = {}
	lmap = load(book='Sanskrit', feat='gpu_lookup')
	# pdb.set_trace()
	if lmap is not None:
		ilmap = dict(zip(lmap.values(), lmap.keys()))
	try:
		print('loading...')
		with open(savepath, "rb") as savefile:
			kwargs = torch.load(savefile)
	except FileNotFoundError:
		print("Not found..")
		kwargs = {
		        "input_size": 32,
		        "output_classes": len(lmap.keys()),
		        "lr": 3e-4,
		        "momentum": 0.8
		}
	engine = Engine(**kwargs)
	print('Training')
	val_err, train_err = engine.train(train_set, debug=True)
	kwargs = engine.export()
	torch.save(kwargs, open(savepath, "wb+"))
	# pdb.set_trace()
	engine.test(test_set)
	decoder  = Decoder(lmap, ilmap)
	for pair in (test_set):
		sequence, target = pair
		prediction = engine.recognize(sequence, decoder)
		print('Prediction: %s'%prediction)
	

if __name__ == '__main__':
	parser = ArgumentParser()
	base_opts(parser)
	args = parser.parse_args()
	config_file = open(args.config)
	config = json.load(config_file)
	transformed_dataset = OCRDataset(config=config, transform=transforms.Compose([ToTensor()]))
	# total = torch.zeros
	total = []
	for i in trange(len(transformed_dataset)):
		sample = transformed_dataset[i]
		total+=(sample)
	train, test = split(total, random=False, split=0.8)
	# pdb.set_trace()
	train_test(train, test)

