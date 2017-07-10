import heapq
from random import randint

def pick_best(**kwargs):
    k = kwargs['count']
    key = kwargs['key']
    def best_k_key(d):
        return heapq.nlargest(k, d.items(), key=key)
    return best_k_key

def sequential(d_item, **kwargs):
	if 'count' in kwargs:
		k = kwargs['count']
	indices, predictions = zip(*d_item)
	k_best = [predictions[index] for index in indices]
	return k_best



    return index

def random_index(d_item):
    return randint(10000)

def word_freqency():


