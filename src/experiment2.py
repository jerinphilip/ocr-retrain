import os
#import pandas as pd
from ocr import GravesOCR
import numpy as np
from error_module import Dictionary
import cv2
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
    pagewise = pagewise[5:10]
    images, truths = [], []
    total,correctable, uncorrectable,correct,real_world_error= 0,0,0,0,0
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
        if error<threshold:
            if truth != prediction:
                real_world_error+=1
            else:
                correct +=1
        else:
            suggestions = cm.suggest(prediction)
            #print(len(suggestions))
            if truth in suggestions:
                correctable+=1
            else:
                uncorrectable+=1
    return([book_path,total,real_world_error, correct, correctable, uncorrectable])



if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))
    book_index = int(sys.argv[2])
    row=[]
    output_dir = '/data5/deepayan/webocr/retrain/'
    ocr = GravesOCR(config["model"], config["lookup"])
    #error = Dictionary(**config["error"])
    #correction = Correction(**config["error"])
    error  = Correction(**config["error"])
    book_locs = list(map(lambda x: config["dir"] + x + '/', config["books"]))
    stat_d = arbitrium(ocr, error, book_locs[book_index])
    #print (stat_d)
    #df = pd.DataFrame(stat_d,columns=['Book_index','Real Word Error','Correct Word','Correctable','Uncorrectable'])
    #df.to_csv('%s/results.csv'%(output_dir))
    with open('%s/results_deep2.txt'%(output_dir),'a+') as fp:
        fp.write('\n'+'Book: '+str(book_index)+'\n'+'Total: '+str(stat_d[1])+'\n'+'real_world_error: '+str(stat_d[2])+'\n'+'correct_word: '+str(stat_d[3])+'\n'+'corectable: '+str(stat_d[4])+'\n'+'uncorrectable: '+str(stat_d[5])+'\n')
