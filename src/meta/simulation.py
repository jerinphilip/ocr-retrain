import sys
import os

# Insert, so root-dir remains clean
src = os.path.abspath("../src")
sys.path.insert(0, src)

from ocr import GravesOCR
from postproc.dictionary import Dictionary
from aux.tokenizer import tokenize
from parser import read_book
import json
from cost_model import CostModel
from timekeep import Timer
from selection import pick_best
from selection import sequential, random_index
from parser.convert import page_to_unit

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
    pagewise = read_book(book_path)
    pagewise = pagewise[:7]
    batchSize = 1

    images, truths = page_to_unit(pagewise)
    n_images = len(images)

    timer.start("ocr, recognize")
    predictions = [ocr.recognize(image) for image in images]
    timer.end()


    running_vocabulary = []

    state_dict = {
            "included": {
                "indices": set()
            },
            "excluded": {
                "indices": set(range(page_count))
            }
    }

    export = {
            "units": n_images,
            "book_dir": book_path,
            "progress": {}
    }

    n_included = 0
    while (n_included < n_images):
        em.enhance_vocabulary(running_vocabulary)
        iter_dict = {}
        timer.start("iteration %d"%(n_included))
        for t in ["included", "excluded"]:
            iter_dict[t] = {}
            indices = state_dict[t]["indices"]
            cost_engine = CostModel(em)
            for i in indices:
                cost_engine.account(predictions[i], truths[i])
            iter_dict[t] = cost_engine.export()

        export["progress"][n_included] = iter_dict

        # Now add batchSize to the list
        batchSize = 1

        fn = pick_best(count=batchSize, key=sequential)
        promoted = fn(iter_dict["excluded"])

        for index, d in promoted:
            running_vocabulary.append(truths[index])
            state_dict["included"]["indices"].add(index)
            state_dict["excluded"]["indices"].remove(index)
            n_included += 1

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


