import doctools.parser.cluster as pc
from doctools.postproc.correction.params import cluster_params as params
from argparse import ArgumentParser
from .debug import time
from .opts import base_opts
from doctools.parser import read_book, text
from doctools.parser.nlp import extract_words
import json
from doctools.cluster.mst import recluster, merge
import doctools.postproc.correction as cost
from doctools.postproc.dictionary import Dictionary

@time
def enhance(book_locs, book_index, em):
    full_text = '\n'.join(list(map(text, book_locs)))
    words = extract_words(full_text)
    em.enhance_vocab_with_books(words)
    return em

if __name__ == '__main__':
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()
    config_file = open(args.config)
    config = json.load(config_file)

    # Load OCR
    print(config["model"])
    #ocr = GravesOCR(config["model"], config["lookup"])
    book_name = config["books"][args.book]
    error_module = Dictionary(**config["error"])
    book_list = config["books"]
    book_locs = list(map(lambda x: config["dir"] + x + '/', book_list))
    error_module = enhance(book_locs, args.book, error_module)

    # Load predictions
    data, status = pc.load(book_name, feat="ocr")
    predictions, truths = data["predictions"], data["truths"]

    _cost, _errors = cost.naive(predictions, truths, error_module)
    print("Naive", _cost, _errors)

    _cost, _errors = cost.suggest(predictions, truths, error_module)
    print("Suggest", _cost, _errors)

    data, status = pc.load(book_name, feat="images", **params["images"])
    ei, ci = data["edges"], data["components"]
    ei, ci = recluster(ei, data["vertices"], threshold=0.18, rep='components')

    _cost, _errors = cost.cluster(predictions, truths, error_module, ci)
    print("Images: ", _cost, _errors)

    data, status = pc.load(book_name, feat="words", **params["words"])
    ew, cw = data["edges"], data["components"]
    #ew, cw = recluster(ew, data["vertices"], =0.1, rep='components')

    _cost, _errors = cost.cluster(predictions, truths, error_module, cw)
    print("Words", _cost, _errors)

    ec, cc = merge(ew, ei, data["vertices"])
    _cost, _errors = cost.cluster(predictions, truths, error_module, cc)
    print("Combined", _cost, _errors)




