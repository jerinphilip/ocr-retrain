from torch.autograd import Variable
import torch
from parser.loader import read_book
from ocr.preproc import DataLoader
import numpy as np
from .dtype import GravesNN
from ocr.pytorch.engine import Engine
from ocr.pytorch.util import gpu_format
from parser.lookup import codebook
from .coding import Decoder
import pdb
book = 'data/Advaita_Deepika/'

def evaluate(model_ft, sequence, target, decoder):
	model_ft.eval()
	sequence = torch.Tensor(np.array(sequence, dtype=np.float32))
	sequence = sequence.unsqueeze(0)
	sequence = sequence.permute(2, 0, 1).contiguous()
	sequence = sequence.cuda()
	sequence = Variable(sequence)
	probs = model_ft(sequence)
	prediction = decoder.decode(probs)
	return prediction
def evaluateRandomly(book, model_ft, decoder):
	pagewise = read_book(book_path=book)
	loader = DataLoader(pagewise=pagewise)
	sequences, targets = loader.sequences, loader.targets
	rand = np.arange(len(sequences))
	np.random.shuffle(rand)
	for i, n in enumerate(rand):
		sequence, target = sequences[n], targets[n]
		print(target)
		output = evaluate(model_ft, sequence, target, decoder)
		print(output)

if __name__ == '__main__':
	savepath = "file.tar"
	lookup_filename = 'lookups/Sanskrit.txt'
	lmap, ilmap = codebook(lookup_filename)
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
	decoder  = Decoder(lmap, ilmap)
	model_ft = engine.model
	evaluateRandomly(book, model_ft, decoder)


