import sys
import os

# Insert, so root-dir remains clean
src = os.path.abspath("../ocr/")
sys.path.insert(0, '../src/')
sys.path.insert(0, '../src/meta/')
from ocr import GravesOCR

from postproc.dictionary import Dictionary
from parser import read_book
import json
from cost_model import CostModel
from timekeep import Timer
from selection import pick_best
from selection import sequential, random_index,  word_frequency
from parser.convert import page_to_unit
import parser.webtotrain as webtotrain
from parser.nlp import extract_words

def simulate(ocr, em, book_locs, book_index):
    timer = Timer(debug=True)
    book_path = book_locs.pop(book_index)
    timer.start("leave one out vocab")
    full_text = '\n'.join(list(map(webtotrain.full_text, book_locs)))
    words = extract_words(full_text)
    em.enhance_vocab_with_books(words)

    timer.start("read images")
    pagewise = webtotrain.read_book(book_path)

    # Comment if main run.
    #num_pages = min(len(pagewise), 10)
    #pagewise = pagewise[:num_pages]

    page_count = len(pagewise)
    batchSize = 500
    images, truths = page_to_unit(pagewise)
    n_images = len(images)
    timer.start("ocr, recognize")
    predictions = [ocr.recognize(image) for image in images]
    timer.end()

    strategies = [
        ("random", random_index),
        ("sequential", sequential),
        ("frequency", word_frequency)
    ]

    export = {
            "units": n_images,
            "book_dir": book_path,
    }

    for strategy, fn in strategies:
        progress = {}
        state_dict = {
                "included": {
                    "indices": set()
                },
                "excluded": {
                    "indices": set(range(n_images))
                }
        }

        n_words_included = 0
        running_vocabulary =[]
        while n_words_included < n_images:
            em.enhance_vocabulary(running_vocabulary)
            iter_dict = {}
            excluded_sample = []

            for i in state_dict["excluded"]["indices"]:
                metric = (i, predictions[i])
                excluded_sample.append(metric)
            
            promoted =  fn(excluded_sample, count=batchSize)
            state_dict["promoted"] = {}
            state_dict["promoted"]["indices"] = promoted

            timer.start("iteration %d"%(n_words_included))
            print('number of words included %d'%(n_words_included))
            #for t in ["included", "excluded", "promoted"]:
            for t in ["excluded", "promoted"]:
                iter_dict[t] = {}
                indices = state_dict[t]["indices"]
                cost_engine = CostModel(em)
                for i in indices:
                    cost_engine.account(predictions[i], truths[i])
                iter_dict[t] = cost_engine.export()

            progress[n_words_included] = iter_dict

            promoted_final = set()
            for meta_index in promoted:
                # One pass to get all similar words
                indices = find_indices(truths, truths[meta_index])
                for index in indices:
                    try:
                        state_dict["included"]["indices"].add(index)
                        state_dict["excluded"]["indices"].remove(index)
                    except KeyError:
                        #print("Not found key", index)
                        pass
                    promoted_final.add(index)
                
            state_dict["promoted"]["scanned_indices"] = promoted
            n_words_included += len(promoted_final)
        export[strategy] = progress
    return export

def find_indices(ls, key):
    indices = []
    for i, x in enumerate(ls):
        if x == key:
            indices.append(i)
    return indices



if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))
    book_index = int(sys.argv[2])
    lang = sys.argv[3]
    output_dir = 'new_outputs'
    ocr = GravesOCR(config["model"], config["lookup"])
    error = Dictionary(**config["error"])
    book_locs = list(map(lambda x: config["dir"] + x + '/', config["books"]))
    stat_d = simulate(ocr, error, book_locs, book_index)
    with open('outputs/%s/stats_%s.json'%(lang,  config["books"][book_index]), 'w+') as fp:
           json.dump(stat_d, fp, indent=4)
