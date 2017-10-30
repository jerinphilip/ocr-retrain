import sys
import os
from doctools.ocr import GravesOCR
from doctools.postproc.dictionary import Dictionary
from doctools.parser import read_book
import json
#from cost_model import CostModel
#from timekeep import Timer
from doctools.parser.convert import page_to_unit
import doctools.parser.webtotrain as webtotrain
from doctools.parser.nlp import extract_words
from collections import Counter
import pdb
import numpy as np
from time import time
def cost_model(**kwargs):
    tc, sc = 15, 5
    #method = kwargs['method']
    in_dictionary = kwargs['included']
    not_in_dictionary = kwargs['excluded']

    if in_dictionary:
        return tc*not_in_dictionary + sc*in_dictionary
    else:
        return tc*not_in_dictionary

def naive(errors):
    #pdb.set_trace()
    cost = cost_model(excluded= errors, included = None)
    return cost

def spell_check(correctable, uncorrectable):
    cost = cost_model(excluded = uncorrectable, included = correctable)
    return cost

def simulate(ocr, em, book_locs, book_index):
    book_path = book_locs.pop(book_index)
    full_text = '\n'.join(list(map(webtotrain.full_text, book_locs)))
    print('Adding words to Vocabulary...')
    t0 = time()
    words = extract_words(full_text)
    em.enhance_vocab_with_books(words)
    print('done in %.2fs.' % (time() - t0))
    print('extracting words and images for the given book')
    t0 = time()
    pagewise = webtotrain.read_book(book_path)
    page_count = len(pagewise)
    images, truths = page_to_unit(pagewise)
    print('done in %.2fs.' % (time() - t0))
    
    n_images = len(images)
    #timer.start("ocr, recognize")
    print('recognizing....')
    t0 = time()
    predictions = [ocr.recognize(image) for image in images]
    print('done in %.2fs.' % (time() - t0))
    
    errors = [em.error(prediction) for prediction in predictions]
    print('calculating the cost')
    t0 = time()
    non_zero = np.count_nonzero(np.array(errors))
    cost_naive = naive(non_zero)
    error_indices = [i for i,v in  enumerate(errors) if v != 0]
    correctable, uncorrectable = 0, 0
    vocab = []
    for index in error_indices:
        em.enhance_vocabulary(vocab)
        suggestions = em.suggest(predictions[index])
        if truths[index] in suggestions:
            correctable += 1
            vocab.append(truths[index])
        else:
            uncorrectable += 1
    
    cost_spellcheck = spell_check(correctable, uncorrectable)
    print('done in %.2fs.' % (time() - t0))
    
    return cost_naive, cost_spellcheck, n_images
    
    
    
if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))
    book_index = int(sys.argv[2])
    lang = sys.argv[3]
    output_dir = 'postproc_analysis'
    ocr = GravesOCR(config["model"], config["lookup"])
    error = Dictionary(**config["error"])
    book_locs = list(map(lambda x: config["dir"] + x + '/', config["books"]))
    print('starting the process')
    t = time()
    cost_naive, cost_spellcheck, n_words = simulate(ocr, error, book_locs, book_index)
    #pdb.set_trace()
    book_name = config["books"][book_index]
    with open('cost.txt', 'a') as in_file:
        in_file.write('\n Book: %s \n cost for Naive Method is %.2f \n cost for spell checker method is %.2f \n number of words processed is %d \n'
        %(book_name,cost_naive, cost_spellcheck, n_words)) 

    print('Process finished in %.2fs.' % (time() - t))
    
    
