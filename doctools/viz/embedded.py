from collections import defaultdict
from pprint import pprint
import sys, random
from nltk.tokenize import RegexpTokenizer
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

def embedded(truths, predictions, components):
    """
    Takes in ground truths, predictions and components.
    Produces a stacked bar chart containing the following information.

    See:
    https://matplotlib.org/examples/statistics/histogram_demo_multihist.html

    """

    # For each prediction, assign it to a cluster.
    # The inverse mapping has to be generated.
    counter = 0
    clusterId = {}
    for component in components:
        print(component)
        for index in component:
            pred = predictions[index]
            clusterId[pred] = counter
        counter = counter + 1


    # Each truth consists of clusters.
    # IR[truth] = (cluster, count)

    # IR = Intermediate Representation
    # Organizes to embed predictions in ground truths.
    # As a stacked histogram.
    IR = defaultdict(lambda : defaultdict(int))
    for i, truth in enumerate(truths):
        pred = predictions[i]
        Id = clusterId[pred]
        IR[truth][pred] += 1

    for base in IR:
        print(base)
        for pred, count in IR[base].items():
            print('\t', pred, count)


    # xs = anything
    # y = length of stacked bars for a given x
    xs = []
    ys = []
    max_length = 0
    for i, base in enumerate(IR):
        xs.append(i)
        y = []
        for pred, count in IR[base].items():
            y.append(count)
        max_length = max(max_length, len(y))
        ys.append(y)

    for i, y in enumerate(ys):
        while len(ys[i]) < max_length:
            ys[i].append(0)
        print(y, ys[i])

    entries = np.array(list(zip(*ys)))
    width = 0.35
    indices = range(len(ys))
    cumulative = np.zeros(len(indices))

    cmap = plt.get_cmap('Paired', max_length)    # PiYG
    colors = []

    for i in range(cmap.N):
        rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
        color = matplotlib.colors.rgb2hex(rgb)
        colors.append(color)

    for i, entry in enumerate(entries):
        p1 = plt.bar(indices, entry, width, color=colors[i], bottom=cumulative)
        cumulative += entry
        print(entry)
    
    plt.show()
        
    # Draw a stacked bar chart, for a given set of xs and ys.
    #ax1.hist(x, n_bins, normed=1, histtype='bar', stacked=True)
    #ax1.set_title('stacked bar')




if __name__ =='__main__': 
    def create_errors(word):
        if random.random() < 0.5:
            return [word]
        else:
            return random.sample(list(edits1(word)), random.randint(3, 10))

    def edits1(word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return deletes + transposes + replaces + inserts

    with open(sys.argv[1]) as fp:
        tokenizer = RegexpTokenizer('[a-zA-z0-9]+')
        text = fp.read().lower()
        tokens = tokenizer.tokenize(text)
        tokens = random.sample(tokens, 5)
        truths, predictions = [], []
        for token in tokens:
            mix = create_errors(token) + [token]*random.randint(1, 5)
            truths.extend([token]*len(mix))
            predictions.extend(mix)

        clusterId = defaultdict(list)
        for i, prediction in enumerate(predictions):
            clusterId[prediction].append(i)

        components = clusterId.values()
        embedded(truths, predictions, components)



    
