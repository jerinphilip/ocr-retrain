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
import json
import pickle
def check_suggestion(error_indices, truths, predictions, em):
    correctable, uncorrectable = 0, 0
    
    for index in error_indices:
            
        suggestions = em.suggest(predictions[index])
        pdb.set_trace()
        if truths[index] in suggestions:
            correctable += 1
            
        else:
            uncorrectable += 1
            em.enhance_vocabulary(truths[index])
    return correctable, uncorrectable

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

def batchCorrect(data, truths, predictions, em):
    error_indices=[]
    cost = 0
    for key, values in data.items():
        error_indices_percluster = []
        try:
            for i in range(len(values)):
                if em.error(predictions[int(values[i])]):
                    error_indices_percluster.append(int(values[i]))

            error_indices.append(error_indices_percluster)
           
        except IndexError:
            print("error... closing")
    un, co, vocab = 0, 0, []
    for error_percluster in error_indices:
        if error_percluster:
            correctable, uncorrectable = check_suggestion(error_percluster, truths, predictions, em)
            un+=uncorrectable
            co+=correctable

            cost += cost_model(excluded = uncorrectable, included = correctable)
    
    return cost


def simulate(ocr, em, book_locs, json_locs, book_index, book_name):
    book_path = book_locs.pop(book_index)
    json_path = json_locs[book_index]
    # pdb.set_trace()
    outpath_pickled = os.path.join('doctools/outdir', 'pickled/')
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
    print('finding the components in each cluster...')
    t0 = time()
    try:
        with open(json_path, 'r') as json_data:
            data = json.load(json_data)
    except TypeError:
        print('%s/%s.json'%(path,book))
    print('done in %.2fs.' % (time() - t0))
    n_images = len(images)
    
    print('recognizing....')
    t0 = time()
    if os.path.exists(os.path.join('%s'%outpath_pickled,'%s.pkl'%book_name)):
        print('Loading predictions...')
        with open(os.path.join('%s'%outpath_pickled,'%s.pkl'%book_name), 'rb') as f:
            predictions = pickle.load(f)
    else:
            predictions = [ocr.recognize(image) for image in images]
    print('done in %.2fs.' % (time() - t0))
    
    errors = [em.error(prediction) for prediction in predictions]
    print('calculating the cost')
    t0 = time()
    non_zero = np.count_nonzero(np.array(errors))
    # cost_naive = naive(non_zero)
    error_indices = [i for i,v in  enumerate(errors) if v != 0]
    vocab = []
    # correctable, uncorrectable = check_suggestion(error_indices, truths, predictions, em, vocab)
    # cost_spellcheck = spell_check(correctable, uncorrectable)
    cost_batch = (batchCorrect(data, truths, predictions, em))
    print('done in %.2fs.' % (time() - t0))

    return cost_naive, cost_spellcheck, cost_batch, n_images
    
    
    
if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))
    book_index = int(sys.argv[2])
    lang = sys.argv[3]
    output_dir = 'postproc_analysis'
    ocr = GravesOCR(config["model"], config["lookup"])
    error = Dictionary(**config["error"])
    book_list = ['0022', '0029','0040','0060','0061','0069', '0191', '0211']
    # book_locs = list(map(lambda x: config["dir"] + x + '/', config["books"]))
    book_locs = list(map(lambda x: config["dir"] + x + '/', book_list))
    json_locs = list(map(lambda x: 'doctools/outdir/jsons/%s.json'%x, book_list))
    print('starting the process')
    t = time()
    book_name = book_list[book_index]
    cost_naive, cost_spellcheck, cost_batch, n_words = simulate(ocr, error, book_locs, json_locs, book_index, book_name)
    #pdb.set_trace()
    text = '\n Book: %s \n cost for Naive Method is %.2f \n cost for spell checker method is %.2f \n cost for clustering method is %.2f \n of words processed is %d \n'%(book_name,cost_naive, cost_spellcheck, cost_batch, n_words)
    with open('cost.txt', 'a') as in_file:
        in_file.write(text) 

    print('Process finished in %.2fs.' % (time() - t))
    
    
