import heapq
from random import randint
from collections import Counter
import random
import pdb
def pick_most_common(predictions,k):
	wordcount = Counter()
	for prediction in predictions:
		wordcount.update([prediction])

	most_common_words = [ww[0] for ii, ww in enumerate(wordcount.most_common(k))]
	return most_common_words

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

def word_freqency(d_item, **kwargs):
	try:
		indices, predictions = zip(*d_item)

		indices, predictions = list(indices), list(predictions)
		print (len(predictions))
		print(predictions[:10])
		k = kwargs['count']
		most_common_words = pick_most_common(predictions,k)

		#print (most_common_words)
		k_best = [predictions.index(each_word) for each_word in most_common_words]
		return k_best
	except ValueError as e:
		print('Not enough argument')


def word_frequency_v02(d_item, **kwargs):
	k = kwargs['count']
	k_best = []
	try:
		indices, predictions = zip(*d_item)
		indices, predictions = list(indices), list(predictions)
		rev_d_item = dict(zip(predictions, indices))
		most_common_words = pick_most_common(predictions,k)
		#k_best = [rev_d_item[each_word] for each_word in most_common_words](
		def subset(pos):
			sub = [indices[p] for p in pos]
			return sub
		for word in most_common_words:
			pos = [i for i,v in enumerate(predictions) if v == word]
			k_best.extend(subset(pos))
		return(k_best)
	except ValueError as e:
		print('Not enough argument')