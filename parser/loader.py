import os
import subprocess
import glob
import numpy as np
import cv2
import sys
import shutil
import pdb
import re
from tqdm import *

class UnequalLength(BaseException):
	pass
def image_prep(image_path):
	image = cv2.imread(image_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return thresh1

def parse_data(path):
	file_paths = list(map(lambda f: path + f, os.listdir(path)))
	def clean(base_name):
		return re.findall(r'\d+', base_name)[-1] 

	def read(text_file):
		with open(text_file, 'r') as f:
			text = f.read()
		return text
	if path.endswith('Images/'): 
		content = {clean(os.path.basename(x)):image_prep(x) for x in file_paths}
		return content
	content = {clean(os.path.basename(x)):read(x) for x in file_paths}
	return content
def images_and_truths(image, plot, text):
	def extract_units(unit):
		x1, y1, w, h = unit
		x2 = x1+w; y2 = y1+h
		line_crop = image[int(y1):int(y2), int(x1):int(x2)]
		newHeight = 32
		aspectRatio = (float(int(x2) - int(x1)) / float(int(y2) - int(y1)))
		newWidth = int(np.ceil(aspectRatio * newHeight))
		try:
			resized_image = cv2.resize(line_crop, (int(newWidth), int(newHeight)), 
				interpolation=cv2.INTER_AREA)
			return np.array((resized_image/255), dtype='uint8')
		except Exception as e:
			print (int(newWidth), int(newHeight))
			print (str((x2 - x1)) + '\n' + str(y2 - y1))
	li = plot.split()
	units = [li[i:i+4] for i in range(0, len(li), 4)]
	unitImages = list(map(extract_units, units))
	unitTruths = [s.strip() for s in text.splitlines()]
	return unitImages , unitTruths

def read_book(**kwargs):
	pairwise =[]
	book_path = kwargs["book_path"]
	dirs = lambda f: os.path.join(book_path, f)
	folder_paths = map(dirs, ['Images/', 'Annotations/', 'Segmentations/'])
	images, text, plots = list(map(parse_data, folder_paths))
	keys = [key for key in images.keys()]
	pbar = tqdm(keys[:100])
	tsize, pno =[], []
	for key in pbar:	
		try:
			pbar.set_description("Processing %s" % key)
			unitImages, unitTruths = images_and_truths(images[key], plots[key], text[key])
			if len(unitImages) != len(unitTruths):
				raise UnequalLength
			pairwise.append([unitImages, unitTruths])
		except Exception:
			print('\n Key does not exist')
			print(key)
		except UnequalLength:
			print('\n Unequal lengths')
			print(key)
	return pairwise	
	
# book = 'data/Advaita_Deepika/'
# pagewise = read_book(book_path=book)