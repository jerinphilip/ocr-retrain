import doctools.parser.cluster as pc
from doctools.postproc.correction.params import cluster_params as params
from argparse import ArgumentParser
from .debug import time
from .opts import base_opts
from doctools.parser import read_book, text
from doctools.parser.nlp import extract_words
import json
from doctools.cluster.mst import recluster, merge, intersection, bugfixcomponents
#import doctools.postproc.correction as cost
import doctools.simulate.correction as cost
from doctools.postproc.dictionary import Dictionary

# cost.naive = lambda *a, **kw: (0, 0)
# cost.suggest = lambda *a, **kw: (0, 0)
# cost.cluster = lambda *a, **kw: (0, 0)

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

    CE = []
    for book_name in config["books"]:
        try:
            # Load predictions
            data, status = pc.load(book_name, feat="ocr")
            predictions, truths = data["predictions"], data["truths"]

            # Naive computation
            _cost, _errors = cost.naive(predictions, truths, error_module)
            print("Naive", _cost, _errors)
            entry = (book_name, _cost, _errors, "naive")
            CE.append(entry)


            # Suggest Technique
            _cost, _errors = cost.suggest(predictions, truths, error_module)
            print("Suggest", _cost, _errors)
            entry = (book_name, _cost, _errors, "suggest")
            CE.append(entry)

            # Non relaxed distane.
            data, status = pc.load(book_name, feat="non_approx")
            c_non_approx = data["components"]
            _cost, _errors = cost.cluster(predictions, truths, error_module, c_non_approx)
            print("Non-approx:", _cost, _errors)
            entry = (book_name, _cost, _errors, "non-approx")
            CE.append(entry)

            # Images Features

            entry = None
            bei, bci = None, None
            for t in range(36, 0, -1):
                data, status = pc.load(book_name, feat="images", **params["images"])
                ei, ci = data["edges"], data["components"]
                ci = bugfixcomponents(ci, data["vertices"])
                threshold = t/100;
                ei, ci = recluster(ei, data["vertices"], threshold=threshold, rep='components')

                _cost, _errors = cost.cluster(predictions, truths, error_module, ci)
                if (entry is None) or (entry[2] < _cost):
                    print("Images: ", _cost, _errors, t)
                    bei, bci = ei, ci
                    entry = (book_name, _cost, _errors, "images")

            CE.append(entry)


            # Word Features
            entry = None
            bew, bcw = None, None
            for t in range(60, 0, -1):
                data, status = pc.load(book_name, feat="words", **params["words"])
                ew, cw = data["edges"], data["components"]
                cw = bugfixcomponents(cw, data["vertices"])
                threshold = t/100;
                ew, cw = recluster(ew, data["vertices"], threshold=threshold, rep='components')
                print(len(ew))

                _cost, _errors = cost.cluster(predictions, truths, error_module, cw)
                if (entry is None) or (entry[2] < _cost):
                    print("Words", _cost, _errors, t)
                    bew, bcw = ew, cw
                    entry = (book_name, _cost, _errors, "images")

            CE.append(entry)


            # Combined features
            ec, cc = merge(bew, bei, data["vertices"])
            cc = bugfixcomponents(cc, data["vertices"])
            _cost, _errors = cost.cluster(predictions, truths, error_module, cc)
            print("Combined", _cost, _errors)
            entry = (book_name, _cost, _errors, "+union")
            CE.append(entry)

            # Combined-Intersection features
            ec, cc = intersection(ew, ei, data["vertices"])
            _cost, _errors = cost.cluster(predictions, truths, error_module, cc)
            print("Intersection", _cost, _errors)
            entry = (book_name, _cost, _errors, "intersection")
            CE.append(entry)
        except Exception as e:
            print("Book", book_name, "failed")
            print(e)
            raise e


    #  es = []
    #  ess = []
    #  for e in ec:
    #      f = lambda e, e1, e2: (e in e1) and (e not in e2)
    #      g = lambda e, e1, e2: (e in e1) and (e in e2)
    #      if f(e, ei, ew):
    #          es.append(e)
    #      if g(e, ei, ew):
    #          ess.append(e)
    #      
    #  print(len(es), len(ess))
    #  print(len(ei), len(ew))
