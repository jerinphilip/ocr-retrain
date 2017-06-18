import numpy as np

def concat(seq_targ_1, seq_targ_2):
    """ 
    Rather than doing batches, do a time series concatenation.
    Leave a blank something in between, so that there are no issues.
    
    """
    seq_1, targ_1 = seq_targ_1
    seq_2, targ_2 = seq_targ_2

    H1, W1 = seq_1.shape
    H2, W2 = seq_1.shape

    assert(H1 == H2)
    w_blank = H1 # 
    blank = np.ones((H1, w_blank), dtype=np.uint8)
    seq = np.concatenate((seq_1, blank, seq_2), axis=1)
    target = targ_1 + ' ' + targ_2

    return (seq, target)


def knot(ls, **kwargs):
    max_width = 4096
    if 'max_width' in kwargs:
        max_width = kwargs['max_width'] 


    if not ls:  return []

    get_width = lambda x: x[0].shape[1]
    knotted = [ls[0]]
    width = get_width(ls[0]) 
    current = 0
    for i in range(1, len(ls)):
        next_width = get_width(ls[i])
        if width + next_width < max_width:
            knotted[current] = concat(knotted[current], ls[i])
            width += next_width
        else:
            knotted.append(ls[i])
            current = current + 1
            width = next_width 

    return knotted
