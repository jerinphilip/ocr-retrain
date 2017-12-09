from doctools.parser.convert import page_to_unit
from doctools.parser import read_book
from argparse import ArgumentParser
from pprint import pprint
from doctools.ocr import GravesOCR
import os
from .opts import base_opts
from .debug import time
import doctools.parser.cluster as pc
import json


@time
def compute(fpath):

    @time
    def read():
        pagewise = read_book(book_path=fpath, unit='word')
        images, truths = page_to_unit(pagewise)
        return (images, truths)

    @time
    def predict(images, truths):
        predictions = ocr.predict(images)
        errored = [i for i in range(len(truths)) if predictions[i] != truths[i]]
        return (predictions, errored)

    images, truths = read()
    predictions, errored = predict(images, truths)

    save_data = {
        "truths": truths,
        "predictions": predictions,
        "errored": errored
    }

    return save_data



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

    outpath = args.output
    outpath_pickled = os.path.join(args.output, 'pickled')

    fpath = os.path.join(config["dir"], book_name)
    data = compute(fpath)
    pc.save(data=data, feat='ocr', book=book_name, params={})
