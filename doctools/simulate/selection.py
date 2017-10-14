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
    k = min(k, len(set(predictions)))
    running_set = set()
    k_best = []
    for index, prediction in d_item:
        if prediction not in running_set:
            running_set.add(prediction)
            k_best.append(index)
            if len(k_best) == k:
                break
    return k_best


def random_index(d_item, **kwargs):
    k = kwargs['count']
    indices, predictions = zip(*d_item)
    k = min(k, len(set(predictions)))
    k_best = []
    running_set = set()
    indices = set(indices)
    d_item = set(d_item)
    while len(k_best) < k:
        index, prediction = random.choice(list(d_item))
        d_item.remove((index, prediction))
        if prediction not in running_set:
            running_set.add(prediction)
            k_best.append(index)
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


        

