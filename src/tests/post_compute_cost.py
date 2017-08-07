import json
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


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



def get_xy(d):
    counts = d["progress"]
    xs, ys = [], []
    review_cost = 0
    cost_per_unit = 0
    batchSize = 20
    for n_included in sorted(counts.keys(), key=lambda x: int(x)):
        state = counts[n_included]
        x = int(n_included)
        x_ = d["units"] - x
        unseen_cost = tree_reduce(state["excluded"], cost)
        cost_per_unit =  unseen_cost / x_
        projected = review_cost + unseen_cost
        xs.append(x)
        print(x, projected, cost_per_unit)
        ys.append(projected)
        review_cost += cost_per_unit * batchSize
    return (xs, ys)

def get_xys(fns):
    pairs = []
    for fn in fns:
    return pairs
        
        
if __name__ == '__main__':
    d = parse(sys.argv[1])
    xs, ys = get_xy(d)


exit()
#print('Lang,Book,Cost,Error,Word')
#for lang in ['hi', 'ml']:
#    saves = []
#    for dr, drs, fls in os.walk('output/%s'%(lang)):
#        for fn in fls:
#            fn_with_path = dr + '/' + fn
#            saves.append(json.load(open(fn_with_path)))
#
#    # words = list(map(get_total_words, saves))
#    plt.figure(figsize=(20,10))
#    for save in saves:
#        xs, ys = get_xs_ys(save)
#        bname = fmap[lang][extract_bcode(save)]
#        plt.plot(xs, ys, label=bname)
#        label_str = "%s, %s"%(bname[:5], save["pages"])
#        plt.text(xs[-1], ys[-1], label_str)
#
#    plt.xlabel("no of pages included in dictionary")
#    plt.ylabel("estimated cost for entire book")
#    plt.savefig('output/images/cost-projected-%s.png'%(lang), dpi=300)
#    plt.clf()
#

