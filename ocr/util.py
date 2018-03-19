import numpy as np
import Levenshtein as lev
from operator import eq

# def concat(seq_targ_1, seq_targ_2, **kwargs):
#     """ 
#     Rather than doing batches, do a time series concatenation.
#     Leave a blank something in between, so that there are no issues.
    
#     """
#     seq_1, targ_1 = seq_targ_1
#     seq_2, targ_2 = seq_targ_2

#     H1, W1 = seq_1.shape
#     H2, W2 = seq_1.shape


#     assert(H1 == H2)
#     w_blank = H1
#     if 'space_width' in kwargs:
#         w_blank = kwargs['space_width']
#     blank = np.ones((H1, w_blank), dtype=np.uint8)
#     seq = np.concatenate((seq_1, blank, seq_2), axis=1)
#     target = targ_1 + ' ' + targ_2

#     return (seq, target)


# def knot(ls, **kwargs):
#     max_width = 4096
#     if 'max_width' in kwargs:
#         max_width = kwargs['max_width'] 

#     # Subroutine to knot.
#     if not ls:  return [] # Base case.
#     get_width = lambda x: x[0].shape[1] # Helper Function - Don't repeat.

#     # Initialization
#     knotted = [ls[0]]
#     current = 0
#     width = get_width(ls[0]) 

#     space_width = 32

#     # Invariant: ?
#     for i in range(1, len(ls)):
#         next_width = get_width(ls[i])
#         if width + space_width + next_width < max_width:
#             knotted[current] = concat(knotted[current], ls[i], 
#                     space_width=space_width)
#             width += space_width + next_width  
#         else:
#             knotted.append(ls[i])
#             current = current + 1
#             width = next_width

#     return knotted


def cer(words, truths):
    sum_edit_dists = sum(map(lev.distance, words, truths))
    sum_gt_lengths = sum(map(len, truths))
    fraction = sum_edit_dists/sum_gt_lengths
    percent = fraction*100
    return percent

def wer(words, truths):
    correct_count = sum([1 if word == truth else 0 for word, truth in zip(words, truths)])
    # correct = filter(map(eq, words, truths))
    # correct_count = len(correct)
    total = len(truths)
    fraction =  correct_count/total
    percent = fraction*100
    return (100.0-percent)
