import json
import sys


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
        
if __name__ == '__main__':
    d = parse(sys.argv[1])
    xs, ys = get_xy(d)


