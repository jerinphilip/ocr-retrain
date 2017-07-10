import heapq
from random import randint
from collections import Counter
import random

def pick_most_common(predictions):
	wordcount = Counter()
	for prediction in predictions:
		wordcount.update(prediction)
	most_common = [ww[0] for ii, ww in enumerate(wordcount.most_common(10000))]
	return most_common

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
	k_best  = random.sample(k, indices)
	return k_best

def word_freqency(d_item, **kwargs):
	indices, predictions = zip(*d_item)
	indices, predictions = list(indices), list(predictions)
	k = kwargs['count']
	most_common = pick_most_common(predictions)
	k_best = [predictions.index[each_word] for each_word in most_common[:k]]
	return k_best



