import os
from ocr import GravesOCR
import numpy as np
from error_module.dictionary import Dictionary
import sys
from aux.tokenizer import tokenize
import parser.webtotrain as webtotrain
from random import randint
import json
from pprint import pprint
from functools import partial, reduce
from math import floor,ceil
from operator import add
from cost_model import CostModel
from timer import Timer
from selection import pick_best
from selection import sequential, random_index

def extract_words(text):
    tokens = []
    lines = text.splitlines()
    for line in lines:
        candidates = tokenize(line)
        valid = filter(lambda x:x, candidates)
        tokens.extend(valid)
    return tokens

def stats(ocr, em, book_locs, book_index):
    timer = Timer(debug=True)
    book_path = book_locs.pop(book_index)

    timer.start("leave one out vocab")
    full_text = '\n'.join(list(map(webtotrain.full_text, book_locs)))
    words = extract_words(full_text)
    em.enhance_vocab_with_books(words)

    timer.start("read images")
    pagewise = webtotrain.read_book(book_path)
    pagewise = pagewise[:7]
    page_count = len(pagewise)
    batchSize = 1

    timer.start("ocr, recognize")
    predictions = [[ocr.recognize(image) 
                        for image in page_images] 
                        for page_images,_ in pagewise]

    timer.end()

    running_vocabulary = []

    state_dict = {
            "included": {
                "indices": set()
            }
            "excluded": {
                "indices": set(range(page_count))
            }
    }

    export = {
            "pages": page_count,
            "book_dir": book_path,
            "progress": {}
    }

    n_included = 0
    while (n_included < page_count):
        iter_dict = {}
        timer.start("iteration %d"%(n_included))
        for t in ["included", "excluded"]:
            iter_dict[t] = {}
            indices = state_dict[t]["indices"]
            for i in indices:
                cost_engine = CostModel(em)
                images, truths = pagewise[i]
                preds = predictions[i]

                for pred, truth in zip(preds, truths):
                    cost_engine.account(pred, truth)
                iter_dict[t][i] = cost_engine.export()

        export["progress"][n_included] = iter_dict

        # Now add batchSize to the list
        batchSize = 1

        fn = pick_best(count=batchSize, key=sequential)
        promoted = fn(iter_dict["excluded"])

        state_dict["included"]["indices"].add(promoted)
        state_dict["excluded"]["indices"].remove(promoted)

    return export

if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))
    book_index = int(sys.argv[2])
    lang = sys.argv[3]
    output_dir = 'srm-27-june'
    ocr = GravesOCR(config["model"], config["lookup"])
    error = Dictionary(**config["error"])
    book_locs = list(map(lambda x: config["dir"] + x + '/', config["books"]))
    stat_d = stats(ocr, error, book_locs, book_index)
    with open('%s/%s/stats_%s.json'%(output_dir, lang, config["books"][book_index]), 'w+') as fp:
            json.dump(stat_d, fp, indent=4)


