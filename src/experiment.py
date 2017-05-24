import os
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

def split_index(data, fraction):
    assert(0 <= fraction and fraction <= 1)
    total = len(data)
    num_first = floor(fraction*total)
    num_second = total - num_first
    print("Num First, Num Second: ", num_first, num_second)
    #print(data[:num_first])
    split_index = 0
    for i in range(num_first):
        split_index += len(data[i][0])
    print("Split Index:", split_index)
    return split_index

def stats(ocr, em, book_path):
    pagewise = webtotrain.read_book(book_path)
    #pagewise = pagewise[5:10]
    images, truths = [], []
    for imgs, ts in pagewise:
        images.extend(imgs)
        truths.extend(ts)

    print("Recognizing..", flush=True)
    predictions = [ocr.recognize(image) for image in images]

    stat_d = {}

    for fraction in [0.2, 0.4, 0.6, 0.8]:
        si = split_index(pagewise, fraction)
        em.enhance_vocabulary(truths[:si])
        print("Computing Errors, with fraction %.1f"%(fraction), flush=True)
        errors = [em.error(prediction) for prediction in predictions]

        initial_dict = lambda : {
                "real_word_error": 0,
                "correctable": 0,
                "correct": 0,
                "uncorrectable":0,
                "total": 0
            }


        sfd = {
            "included": initial_dict(),
            "unseen": initial_dict()
        }
        threshold = 1
        for i in range(len(truths)):
            truth, prediction, error = truths[i], predictions[i], errors[i]
            dkey = lambda dtype: "included" if dtype else "unseen"
            sfd[dkey(i<si)]["total"] += 1
            if error < threshold:
                if truth != prediction:
                    sfd[dkey(i<si)]["real_word_error"] += 1
                else:
                    sfd[dkey(i<si)]["correct"] += 1
            else:
                suggestions = em.suggest(prediction)
                if truth in suggestions:
                    sfd[dkey(i<si)]["correctable"] += 1
                else:
                    sfd[dkey(i<si)]["uncorrectable"] += 1

        stat_d[fraction] = sfd

    stat_d["book_dir"] = book_path
    return stat_d

if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))
    book_index = int(sys.argv[2])
    output_dir = 'output'
    ocr = GravesOCR(config["model"], config["lookup"])
    error = Dictionary(**config["error"])
    book_locs = list(map(lambda x: config["dir"] + x + '/', config["books"]))
    stat_d = stats(ocr, error, book_locs[book_index])
    with open('%s/ml-ocr/stats_%s.json'%(output_dir, config["books"][book_index]), 'w+') as fp:
        json.dump(stat_d, fp, indent=4)
