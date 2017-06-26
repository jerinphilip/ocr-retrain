import heapq
from random import randint

def pick_best(**kwargs):
    k = kwargs['count']
    key = kwargs['key']
    def best_k_key(d):
        return heapq.nlargest(k, d.items(), key=key)
    return best_k_key


def sequential(d_item):
    index, rest = d_item
    return index

def random_index(d_item):
    return randint(10000)


