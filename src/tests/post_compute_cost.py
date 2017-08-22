import json
import sys
import matplotlib
import os
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import fnmatch


# Global cost parameters.

cost = {
        # All units in seconds
        "true": { # Correct by OCR
            "true": 0, # Verified by dict.
            "false": { 
                "true": 0, # Suggestions contain
                "false": 0 # Full type not necessary since GT = OCR Pred
            }
        },
        "false": { # Incorrect by OCR
            "true": 0, # Verified by dict => RWE
            "false": {
                "true": 5, # Suggestions contain
                "false": 16, # Full type time
            }
        }
}

def parse(filename):
    with open(filename) as fp:
        d = json.load(fp)
        return d

def tree_reduce(d, c):
    r = 0
    for key in c:
        if type(c[key]) is dict:
            r += tree_reduce(d[key], c[key])
        else:
            r += c[key]*d[key]
    return r



def get_xy(counts):
    xs, ys = [], []
    review_cost = 0
    previous_cost = 0
    cost_per_unit = 0
    keys = list(filter(lambda x: x.isnumeric(), counts.keys()))
    skeys = sorted(keys, key=lambda x: int(x))
    previous = 0
    for n_included in skeys:
        state = counts[n_included]
        x = int(n_included)
        batchSize = x - previous
        previous = x
        unseen_cost = tree_reduce(state["excluded"], cost)
        included_cost = tree_reduce(state["included"], cost)
        delta_cost = tree_reduce(state["promoted"], cost)
        projected = review_cost + unseen_cost
        review_cost += delta_cost
        print("Review Cost, New cost = %d, %d"%(review_cost, included_cost))
        xs.append(x)
        ys.append(projected)
    return (xs, ys)
        
for lang in ['ml']:
    saves, fnames = [], []
    for dr, drs, fls in os.walk('outputs/%s'%(lang)):
        for fn in fls:
            fnames.append(fn.split('.')[0].split('_')[-1])
            fn_with_path = dr + '/' + fn
            print(fn_with_path)
            saves.append(json.load(open(fn_with_path)))

    plt.figure(figsize=(20,10))
    i=0
    for save in saves:
        bname = fnames[i]
        for method in ["sequential", "frequency", "random"]:
            xs, ys = get_xy(save[method])
            plt.plot(xs, ys, label=method)
            label_str = "%s, %s"%(bname, save["units"])
            #plt.text(xs[-1], ys[-1], label_str)
        plt.legend()
        plt.xlabel("no of words included in dictionary")
        plt.ylabel("estimated cost for entire book")
        plt.savefig('outputs/images/cp-%s-%s.png'%(lang, bname), dpi=300)
        plt.clf()
        i += 1
