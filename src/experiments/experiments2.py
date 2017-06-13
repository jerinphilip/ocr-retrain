import os
#from ocr import GravesOCR
import numpy as np
from error_module import Dictionary
#import cv2
import sys
from aux.tokenizer import tokenize
import parser.webtotrain as webtotrain
from timeit import timeit
from Levenshtein import distance
from random import randint
import json
from pprint import pprint
from functools import partial
from math import floor,ceil
from error_module import Correction

def arbitrium(ocr,cm,book_path):
	pagewise = webtotrain.read_book(book_path)
    #pagewise = pagewise[5:10]
    images, truths = [], []
    total,correctable, uncorrectable = [],[],[]
    for imgs, ts in pagewise:
        images.extend(imgs)
        truths.extend(ts)
    print("Recognizing..", flush=True)
    predictions = [ocr.recognize(image) for image in images]
    print("computing errors",flush=True )
    errors = [cm.error(prediction) for prediction in predictions]
    threshold = 1
    for i in range(len(truths)):
        truth, prediction, error = truths[i], predictions[i], errors[i]
        total+=1
        if error < threshold:
                if truth != prediction:
                	suggestions = cm.edit(prediction)
                	if truth in suggestions:
                		correctable+=1
                	else:
                		uncorrectable+=1
	correctable_percenatge=float(correctable/total)*100.0
	uncorrectable_percentage  = float(uncorrectable/total)*100.0
	return([correctable_percenatge,uncorrectable_percentage])



if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))
    book_index = int(sys.argv[2])
    output_dir = 'output'
    ocr = GravesOCR(config["model"], config["lookup"])
    #error = Dictionary(**config["error"])
    correction = Correction(**config["error"])
    book_locs = list(map(lambda x: config["dir"] + x + '/', config["books"]))
    stat_d = stats(ocr, error, book_locs[book_index])
    with open('/data5/deepayan/webocr/results.txt','a+') as fp:
        fp.write('correctable %:'+stat_d[0]+'uncorrectable_%:'+stat_d[1])