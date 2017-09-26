import sys
import os
import json

# Insert, so root-dir remains clean
from docutils.ocr import GravesOCR
from docutils.postproc.dictionary import Dictionary
from docutils.parser import read_book
from docutils.meta.cost_model import CostModel
from docutils.meta.timekeep import Timer
from docutils.meta.selection import sequential, random_index,  word_frequency
from docutils.parser.convert import page_to_unit
import docutils.parser.webtotrain as webtotrain
from docutils.parser.nlp import extract_words

class Simulator:
    def __init__(self, **kwargs):
        self.ocr = kwargs['ocr']
        self.em = kwargs['postproc']
        self.fpaths = kwargs['books']
        self.batch_size = kwargs['batch_size']
        self.strategies = [
            ("random", random_index),
            ("sequential", sequential),
            ("frequency", word_frequency)
        ]
        self.t = 0
        self.export = {}
        self.initialize_state()

    def initialize_state(self):
        self.state = {}
        for strategy, fn in self.strategies:
            self.state[strategy] = State(strategy=fn, 
                    predictions=self.predictions)

    def leave_one_out(self, index):
        self.index = index
        self.load_vocabulary()

    def load_vocabulary(self):
        words = []
        for i, fpath in enumerate(self.fpaths):
            if i != self.index:
                text = webtotrain.full_text(fpath)
                bwords = extract_words(text)
                words.append(bwords)
        self.em.enhance_vocab_with_books(words)

    def recognize(self):
        fpath = self.fpaths[self.index]
        pagewise = webtotrain.read_book(fpath)
        if self.debug:
            pagewise = pagewise[5:10]
        images, self.truths = page_to_unit(pagewise)
        self.predictions = [ self.ocr.recognize(image) \
                             for image in images ]
        

    def postprocess(self):
        self.export = {}
        for strategy, fn in self.strategies:
            self.vocabulary = []
            self.export[strategy] = {}
            for state in self.state[strategy]:
                delta = state.export()
                print(delta)


class State:
    def __init__(self, **kw):
        self.included = set()
        self.excluded = set(list(range(kw['count'])))
        self.strategy = kw['strategy']
        self.predictions = kwargs['predictions']
        self.best = []
        self.promoted = set()

    def export(self):
        state = {
           # "included": self.included,
           # "excluded": self.excluded,
            "best": self.best,
            "promoted": self.promoted,
        }
        return state

    def __iter__(self):
        return self

    def __next__(self):
        if not self.excluded:
            raise StopIteration()
        self.promote()
        return self

    def promote(self, **kwargs):
        self.pick()
        self.excluded = self.excluded - self.promoted
        self.included = self.included + self.promoted

    def pick(self):
        excluded = []
        for i in self.excluded:
            metric = (i, self.predictions[i])
            excluded.append(metric)
        self.best = self.strategy(excluded, count=self.batch_size)
        self.promoted = set()
        for i in best:
            self.promoted.add(i)
            for j in excluded_indices:
                if self.truths[i] == self.truths[j]:
                    self.promoted.add(j)

if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))
    book_index = int(sys.argv[2])
    lang = sys.argv[3]
    output_dir = 'new_outputs'
    ocr = GravesOCR(config["model"], config["lookup"])
    error = Dictionary(**config["error"])
    book_locs = list(map(lambda x: config["dir"] + x + '/', config["books"]))
    #stat_d = simulate(ocr, error, book_locs, book_index)
    simulation = Simulator(ocr=ocr, postproc=error, books=book_locs)
    simulation.leave_one_out(book_index)
    simulation.recognize()
    simulation.postprocess()


