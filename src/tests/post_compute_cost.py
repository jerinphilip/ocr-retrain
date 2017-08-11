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
        #print(x, projected, cost_per_unit)
        ys.append(projected)
        review_cost += cost_per_unit * batchSize
    return (xs, ys)

#def get_xys(fns):
#    pairs = []
#    for fn in fns:
#    return pairs
        
        
'''if __name__ == '__main__':
    d = parse(sys.argv[1])
    xs, ys = get_xy(d)
    mydict = [(x,y)  for x,y in zip(xs, ys)]
    for each_element in mydict:
        print(str(each_element)+'\n')

#file structure should be 
 #                          path
 #                             |
 #                             |language(hi or ml)
 #                                           |
 #                                           |method folders(sequential, random or wf)


list2=[]
unique_fls = os.listdir(os.path.join(path, 'ml', 'sequential')) or os.listdir(os.path.join(path,'ml', 'wf'))
for each_uniqe_fl in unique_fls:
    for dr, drs, fls in os.walk(os.path.join(path,'ml')):
        if drs:
            
            list1 =[]
            for each_dir in drs:
                list1.append(os.path.join(path, 'ml',each_dir, each_uniqe_fl))

            list2.append(list1)
print(list2)
method = {0: 'sequential',2:'wf'}

for i in range(len(list2)):
    saves=[]
    for each_element in list2[i]:
        
        saves.append(json.load(open(each_element)))
    plt.figure(figsize=(20,10))
    j=0
    for save in saves:
        
        xs, ys = get_xy(save)
        bname = method[j]
        print(bname)
        j+=1
        
        plt.plot(xs, ys, label = bname)
        label_str = "%s, %s"%(bname, save["units"])
        plt.text(xs[-1], ys[-1], label_str)
        
    plt.xlabel("no of words included in dictionary")
    plt.ylabel("estimated cost for entire book")
    plt.savefig('outputs/images/cost-projected-%s.png'%(unique_fls[i].split('/')[-1]), dpi=300)
    plt.clf()
'''
#print('Lang,Book,Cost,Error,Word')
#path = sys.argv[1]
for lang in ['ml']:
    saves, fnames = [], []
    for dr, drs, fls in os.walk('new_outputs/%s/wf'%(lang)):
        for fn in fls:
            fnames.append(fn.split('.')[0].split('_')[-1])
            fn_with_path = dr + '/' + fn
            print(fn_with_path)
            saves.append(json.load(open(fn_with_path)))

    # words = list(map(get_total_words, saves))
    plt.figure(figsize=(20,10))
    i=0
    for save in saves:
        #print(save)
        xs, ys = get_xy(save)
        #bname = fmap[lang][extract_bcode(save)]
        bname = fnames[i]
        plt.plot(xs, ys)
        #label_str = "%s, %s"%(bname[:5], save["pages"])
        label_str = "%s, %s"%(bname, save["units"])
        plt.text(xs[-1], ys[-1], label_str)
        i+=1
    plt.xlabel("no of words included in dictionary")
    plt.ylabel("estimated cost for entire book")
    plt.savefig('new_outputs/%s/wf/cost-projected.png'%(lang), dpi=300)
    plt.clf()


