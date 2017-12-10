from argparse import ArgumentParser
from doctools.ocr import GravesOCR
import os, json
from .opts import base_opts
from .debug import time
import doctools.parser.cluster as pc
from doctools.postproc.correction.params import cluster_params as params

if __name__ == '__main__':
 
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()
    config_file = open(args.config)
    config = json.load(config_file)

    # Load OCR
    print(config["model"])
    ocr = GravesOCR(config["model"], config["lookup"])
    book_name = config["books"][args.book]
    print("Book:", book_name)
    accs = []
    for book_name in config["books"]:
        try:
            fpath = os.path.join(config["dir"], book_name)
            data, status = pc.load(book_name, params={}, feat='ocr')
            errors = len(data["errored"])
            total = len(data["predictions"])
            acc = (total-errors)/total
            accs.append(acc)
        except:
            pass

    print(sum(accs)/len(accs))
