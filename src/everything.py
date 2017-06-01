import os
from ocr import GravesOCR
import numpy as np
from error_module.dictionary import Dictionary
import sys
from aux.tokenizer import tokenize
import parser.webtotrain as webtotrain
from time import time
from random import randint
import json
from pprint import pprint
from functools import partial, reduce
from math import floor,ceil
from operator import add

def split_index(data, num_first):
    #assert(0 <= fraction and fraction <= 1)
    total = len(data)
    split_index = 0
    for i in range(num_first):
        split_index += len(data[i][0])
    print("Split Index:", split_index)
    return split_index

def extract_words(text):
    tokens = []
    lines = text.splitlines()
    for line in lines:
        candidates = tokenize(line)
        valid = filter(lambda x:x, candidates)
        tokens.extend(valid)
    return tokens

def stats(ocr, em, book_locs, book_index):
    book_path = book_locs.pop(book_index)
    full_text = '\n'.join(list(map(webtotrain.full_text, book_locs)))
    words = extract_words(full_text)
    print("Enhancing with books...", flush=True)
    em.enhance_vocab_with_books(words)
    read_start = time()
    print("Reading Images...", flush=True)
    pagewise = webtotrain.read_book(book_path)
    #pagewise = pagewise[5:8]
    page_count = len(pagewise)
    read_finish = time()
    print("Read Images %d pages in %d seconds."%(page_count, read_finish - read_start), flush=True)

    #pagewise = pagewise[5:8]
    images, truths = [], []
    for imgs, ts in pagewise:
        images.extend(imgs)
        truths.extend(ts)

    recog_start = time()
    print("Recognizing...", flush=True)
    predictions = [ocr.recognize(image) for image in images]
    recog_finish = time()
    print("Recognized %d images in %d seconds."%(len(images), recog_finish-recog_start))
    stat_d = {}
    stat_d["pages"] = page_count

    batch_start_time, batch_end_time = None, None
    batchSize = 10
    #batchSize = 1
    for npage_to_include in range(0, page_count, batchSize):
        batch_start_time = time()
        si = split_index(pagewise, npage_to_include)
        em.enhance_vocabulary(truths[:si])
        print("Computing Errors, with %d pages included."%(npage_to_include), flush=True)
        errors = [em.error(prediction) for prediction in predictions]

        initial_dict = lambda : {
                "real_word_error": 0,
                "correctable": 0,
                "correct": 0,
                "uncorrectable":0,
                "total": 0,
                "ocr_equals_gt": {
                    "correctable": 0,
                    "uncorrectable": 0
                }
            }

        sfd = {
            "included": initial_dict(),
            "unincluded": initial_dict()
        }
        threshold = 1
        for i in range(len(truths)):
            truth, prediction, error = truths[i], predictions[i], errors[i]
            dkey = lambda dtype: "included" if dtype else "unincluded"
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
                    if truth == prediction:
                        sfd[dkey(i<si)]["ocr_equals_gt"]["correctable"] += 1
                else:
                    sfd[dkey(i<si)]["uncorrectable"] += 1
                    if truth == prediction:
                        sfd[dkey(i<si)]["ocr_equals_gt"]["uncorrectable"] += 1

        stat_d[npage_to_include] = sfd
        batch_end_time = time()
        print("Finished batch in %d seconds"%(batch_end_time - batch_start_time))

    stat_d["book_dir"] = book_path
    return stat_d

if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))
    book_index = int(sys.argv[2])
    lang = sys.argv[3]
    output_dir = 'output'
    ocr = GravesOCR(config["model"], config["lookup"])
    error = Dictionary(**config["error"])
    book_locs = list(map(lambda x: config["dir"] + x + '/', config["books"]))
    stat_d = stats(ocr, error, book_locs, book_index)
    with open('%s/%s/stats_%s.json'%(output_dir, lang, config["books"][book_index]), 'w+') as fp:
            json.dump(stat_d, fp, indent=4)


