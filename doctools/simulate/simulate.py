import sys
import os
import json
from pprint import pprint
from collections import defaultdict

# Insert, so root-dir remains clean
from doctools.ocr import GravesOCR
from doctools.postproc.dictionary import Dictionary
from doctools.parser import read_book
from doctools.simulate.cost_model import CostModel
from doctools.simulate.timekeep import Timer
from doctools.simulate.selection import sequential, random_index,  word_frequency
from doctools.parser.convert import page_to_unit
import doctools.parser.webtotrain as webtotrain
from doctools.parser.nlp import extract_words

class Simulator:
    def __init__(self, **kwargs):
        self.ocr = kwargs['ocr']
        self.em = kwargs['postproc']
        self.fpaths = kwargs['books']
        self.n_batches = kwargs['batches']
        self.debug = False
        self.strategies = [
            ("random", random_index),
            ("sequential", sequential),
            ("frequency", word_frequency)
        ]
        self.t = 0
        self.export = {}

    def initialize_state(self):
        self.state = {}
        for strategy, fn in self.strategies:
            self.state[strategy] = State(strategy=fn, 
                    predictions=self.predictions, divisions=self.n_batches)

    def leave_one_out(self, index):
        self.index = index
        self.load_vocabulary()

    def load_vocabulary(self):
        words = []
        for i, fpath in enumerate(self.fpaths):
            if i != self.index:
                text = webtotrain.full_text(fpath)
                bwords = extract_words(text)
                words.extend(bwords)
        self.em.enhance_vocab_with_books(words)

    def recognize(self):
        fpath = self.fpaths[self.index]
        print(fpath)
        pagewise = webtotrain.read_book(fpath)
        if self.debug:
            pagewise = pagewise[5:10]
        images, self.truths = page_to_unit(pagewise)
        self.predictions = [ self.ocr.recognize(image) \
                             for image in images ]
        self.initialize_state()
    
    def compute_cost(self, indices):
        costmodel = CostModel(self.em)
        for i in indices:
            costmodel.account(self.predictions[i], self.truths[i])
        #return costmodel.export()
        return costmodel.linear()


    def postprocess(self):
        self.export = {}
        for strategy, fn in self.strategies:
            self.vocabulary = []
            self.export[strategy] = {
                    "cost": defaultdict(list),
                    "index": defaultdict(list)
            }
            for state in self.state[strategy]:
                self.em.enhance_vocabulary(state.vocabulary)
                delta = state.export()
                headers, ecost = self.compute_cost(state.excluded)
                headers, pcost = self.compute_cost(delta["promoted"])

                self.export[strategy]["cost"]["excluded"].append(ecost)
                self.export[strategy]["cost"]["promoted"].append(pcost)

                for key in delta:
                    ls = list(delta[key])
                    self.export[strategy]["index"][key].append(ls)

        return self.export

class State:
    def __init__(self, **kw):
        self.included = set()
        self.strategy = kw['strategy']
        self.predictions = kw['predictions']
        self.batch_size = int(len(self.predictions)/kw['divisions'])
        count = len(self.predictions)
        self.excluded = set(list(range(count)))
        self.best = []
        self.promoted = set()
        self.flag = False
        self.vocabulary = set()

    def export(self):
        state = {
            "best": self.best,
            "promoted": self.promoted,
        }
        return state

    def __iter__(self):
        return self

    def __next__(self):
        if not self.excluded:
            raise StopIteration()
        if self.flag: self.promote()
        else: self.flag = True
        return self

    def promote(self, **kwargs):
        self.pick()
        self.excluded = self.excluded - self.promoted
        self.included = self.included ^ self.promoted
        for i in self.promoted:
            self.vocabulary.add(self.predictions[i])

    def pick(self):
        excluded = []
        for i in self.excluded:
            metric = (i, self.predictions[i])
            excluded.append(metric)
        self.best = self.strategy(excluded, count=self.batch_size)
        self.promoted = set()
        for i in self.best:
            self.promoted.add(i)
            for j in self.excluded:
                if self.predictions[i] == self.predictions[j]:
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
    simulation = Simulator(ocr=ocr, postproc=error, books=book_locs, batch_size=100)
    simulation.leave_one_out(book_index)
    simulation.recognize()
    stats = simulation.postprocess()
    pprint(stats)


