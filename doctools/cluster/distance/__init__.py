from .jaccard import jaccard, inv_jaccard
from distance import levenshtein
from scipy.spatial.distance import euclidean as euc
from scipy.spatial.distance import cosine as cos

from functools import partial

lev = partial(levenshtein, normalized=True)
