import sys
import os

# Insert, so root-dir remains clean
src = os.path.abspath("../ocr/")
sys.path.insert(0, '../src/')
sys.path.insert(0, '../src/meta/')
from ocr import GravesOCR

from postproc.dictionary import Dictionary
from aux.tokenizer import tokenize
from parser import read_book
import json
#from cost_model import CostModel
from timekeep import Timer
from selection import pick_best
from selection import sequential, random_index
from parser.convert import page_to_unit
import parser.webtotrain as webtotrain
import pdb

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
    pagewise = pagewise[:4]
    page_count = len(pagewise)
    batchSize = 20
    images, truths = page_to_unit(pagewise)
    n_images = len(images)
    timer.start("ocr, recognize")
    predictions = [ocr.recognize(image) for image in images]
    timer.end()
    state_dict = {
            "included": {
                "indices": list()
            },
            "excluded": {
                "indices": list(range(n_images))
            }
    }

    export = {
            "units": n_images,
            "book_dir": book_path,
            "progress": {}
    }
    n_words_included = 0
    running_vocabulary =[]
    for n_words_included in range(0, n_images, batchSize):
        em.enhance_vocabulary(running_vocabulary)
        
        
        
        excluded_sample = []
        for i in state_dict["excluded"]["indices"]:
            metric = (i, predictions[i])
            excluded_sample.append(metric)
        print(len(excluded_sample))
        print(excluded_sample[0])
        promoted = sequential(excluded_sample, count= batchSize)
        for index in promoted:
            running_vocabulary.append(truths[index])
            #state_dict["included"]["indices"].add(index)
            #state_dict["excluded"]["indices"].remove(index)
            state_dict["included"]["indices"].append(index)
            state_dict["excluded"]["indices"].remove(index)
    list(set(state_dict["included"]["indices"]))
    list(set(state_dict["excluded"]["indices"]))
    print("done")      
    return state_dict








if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))
    book_index = int(sys.argv[2])
    lang = sys.argv[3]
    output_dir = 'outputs'
    ocr = GravesOCR(config["model"], config["lookup"])
    error = Dictionary(**config["error"])
    book_locs = list(map(lambda x: config["dir"] + x + '/', config["books"]))
    stat_d = stats(ocr, error, book_locs, book_index)
    with open('%s/%s/stats_%s.json'%(output_dir, lang, config["books"][book_index]), 'w+') as fp:
            json.dump(stat_d, fp, indent=4)

