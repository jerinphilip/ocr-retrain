import matplotlib
import os
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot(methods, fname):
    for method, axes in methods:
        plt.plot(*axes, label=method)

    plt.legend()
    plt.xlabel("no of words included in dictionary")
    plt.ylabel("estimated cost for entire book")
    plt.savefig(fname, dpi=300)
    plt.clf()



