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
    k = min(k, len(predictions))
    k_best = [index for index in indices[:k]] 
    return k_best


def random_index(d_item, **kwargs):
    k = kwargs['count']
    indices, predictions = zip(*d_item)
    k = min(k, len(predictions))
    indices = list(indices)
    k = min(len(indices),k)
    k_best  = random.sample(indices, k)
    return k_best


def word_frequency(d_item, **kwargs):
    k = kwargs['count']
    # Create an inverse mapping
    table = {}
    _, predictions = zip(*d_item)
    k = min(k, len(predictions))
    counter = Counter(predictions)
    for i, p in d_item:
        if p not in table:
            table[p] = i
    w_best = counter.most_common(k)
    k_best = [table[w] for w, f in w_best]

    j = 0
    # Commenting out below part, to process in batch.
    #while len(k_best) < k:
    #    i, p = d_item[j]
    #    if table[p] != i:
    #        k_best.append(i)
    #    j = j + 1
    return k_best


        

