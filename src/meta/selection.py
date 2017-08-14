import heapq
from random import randint
from collections import Counter
import random
import pdb

def pick_best(**kwargs):
    k = kwargs['count']
    key = kwargs['key']
    def best_k_key(d):
        return heapq.nlargest(k, d.items(), key=key)
    return best_k_key

def sequential(d_item, **kwargs):
    k = kwargs['count']
    indices, predictions = zip(*d_item)
    k_best = [index for index in indices[:k]] 
    return k_best


def random_index(d_item, **kwargs):
    k = kwargs['count']
    
    indices, predictions = zip(*d_item)
    indices = list(indices)
    k = min(len(indices),k)
    k_best  = random.sample(indices, k)
    return k_best


def word_frequency(d_item, **kwargs):
    k = kwargs['count']
    # Create an inverse mapping
    table = {}
    _, predictions = zip(*d_item)
    counter = Counter(predictions)
    for i, p in d_item:
        if p not in table:
            table[p] = i
    w_best = counter.most_common(k)
    k_best = [table[w] for w in w_best]
    return k_best


        

