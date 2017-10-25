
def jaccard(w1, w2):
    w1 = set(w1)
    w2 = set(w2)
    return float(len(w1 & w2)) / len(w1 | w2)
