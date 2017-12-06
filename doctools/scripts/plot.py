import matplotlib
import os
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np 
import seaborn as sns
from doctools.scripts.opts import base_opts
import json
from doctools.meta.file_locs import get_clusters


def plot(methods, fname):
    for method, axes in methods:
        plt.plot(*axes, label=method)

    plt.legend()
    plt.xlabel("no of words included in dictionary")
    plt.ylabel("estimated cost for entire book")
    plt.savefig(fname, dpi=300)
    plt.clf()



if __name__ == '__main__':
	parser = ArgumentParser()
	base_opts(parser)
	args = parser.parse_args()
	config_file = open(args.config)
	config = json.load(config_file)
	print(config["model"])
	ocr = GravesOCR(config["model"], config["lookup"])
	error_module = Dictionary(**config["error"])
	outpath = args.output