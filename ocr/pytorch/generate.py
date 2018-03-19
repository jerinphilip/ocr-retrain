from torch.autograd import Variable
import torch
from parser.loader import read_book
from ocr.preproc import DataLoader
import numpy as np
from .dtype import GravesNN
from ocr.pytorch.engine import Engine
from ocr.pytorch.util import gpu_format, load
from parser.lookup import codebook
from .coding import Decoder
import pdb

def evaluate(model_ft, decoder):
	def get_output(sequence):
		model_ft.eval()
		sequence = torch.Tensor(np.array(sequence, dtype=np.float32))
		sequence = sequence.unsqueeze(0)
		sequence = sequence.permute(2, 0, 1).contiguous()
		sequence = sequence.cuda()
		sequence = Variable(sequence)
		probs = model_ft(sequence)
		prediction = decoder.decode(probs)
		pdb.set_trace()
		return prediction
	return get_output

def evaluatePagewise(book, model_ft, decoder):
	def ocr_out(page):
		sequences = page[0]
		f = evaluate(model_ft, decoder)
		return list(map(f, sequences))
	def log_out(pno, line):
		with open('%d.txt'%pno, 'a') as fp:
			fp.write(line)
	pagewise, pno = read_book(book_path=book)
	predictions = list(map(ocr_out,pagewise[:2]))
	pdb.set_trace()
	map(log_out, zip(pno,predictions))
if __name__ == '__main__':
	savepath = "file_sanskrit.tar"
	lmap= load(book='Sanskrit', feat='lookup')
	if lmap is not None:
		ilmap = dict(zip(lmap.values(), lmap.keys()))
	# lookup_filename = 'lookups/Sanskrit.txt'
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
	book = 'data/Advaita_Deepika/'
	evaluatePagewise(book, model_ft, decoder)


