import numpy as np


def euclid_norm(v):
    return np.sqrt(v.dot(v))

def normalized_euclid_norm(u,v):
    return euclid_norm((u-v))/2
