import json
import os
import matplotlib
from pprint import pprint
matplotlib.use('Agg')

from matplotlib import pyplot as plt
plt.figure(figsize=(20,10))

def re_x():
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

def generate_required(save):
    book_ps = save["book_dir"].split('/')
    book = book_ps[-2]
    keys = map(lambda x: x/10, range(11))
    skeys = map(str, keys)
    xs, cp, cbp, icbp = [], [], [], []
    for known_percent in skeys:
        #unseen = save[known_percent]["unseen"]
        dkeys = ["correct", "total", "real_word_error", "correctable", "uncorrectable"]
        cumulative = {}
        for dkey in dkeys:
            cumulative[dkey] = 0
            cumulative[dkey] += save[known_percent]["unseen"][dkey]
            cumulative[dkey] += save[known_percent]["included"][dkey]

        correct_percent = cumulative["correct"]/cumulative["total"]

        to_suggest = cumulative["total"] - \
                (cumulative["correct"] + cumulative["real_word_error"])

        correctable_percent = cumulative["correctable"]/to_suggest
        uncorrectable_percent = cumulative["uncorrectable"]/to_suggest
        xs.append(known_percent)
        cp.append(correct_percent)
        cbp.append(correctable_percent)
        icbp.append(uncorrectable_percent)

    return (xs, cp, cbp, icbp, book)

fmap = {}
fmap['hi'] = dict(map(lambda x: x.strip().split('_'), open("hi.fmap")))
fmap['ml'] = dict(map(lambda x: x.strip().split('_'), open("ml.fmap")))
pprint(fmap)

bbanchor = (1,0.5)
dpi = 200

for lang in ['hi', 'ml']:
    saves = []
    for dr, drs, fls in os.walk('output/%s-final'%(lang)):
        for fn in fls:
            fn_with_path = dr + '/' + fn
            saves.append(json.load(open(fn_with_path)))

    values = map(generate_required, saves)
    xs, cps, cbps, icbps, books = zip(*values)

    re_x()
    handles = []
    for x, y, book in zip(xs, cps, books):
        print(book, lang)
        p, = plt.plot(x, y, label=fmap[lang][book])
        handles.append(p)
        plt.xlabel("fraction of data in vocabulary")
        plt.ylabel("correct percent")

    plt.legend(handles=handles, loc='center left', bbox_to_anchor=bbanchor)
    plt.savefig("output/images/all_correct_%s.png"%(lang), dpi=dpi)
    plt.clf()

    re_x()
    handles = []
    for x, y, book in zip(xs, cbps, books):
        p, = plt.plot(x, y, label=fmap[lang][book])
        handles.append(p)
        plt.xlabel("fraction of data in vocabulary")
        plt.ylabel("correctable percent")

    plt.legend(handles=handles, loc='center left', bbox_to_anchor=bbanchor)
    plt.savefig("output/images/all_correctable_%s.png"%(lang), dpi=dpi)
    plt.clf()

    re_x()
    handles = []
    for x, y, book in zip(xs, icbps, books):
        p, = plt.plot(x, y, label=fmap[lang][book])
        handles.append(p)
        plt.xlabel("fraction of data in vocabulary")
        plt.ylabel("uncorrectable percent")

    plt.legend(handles=handles, loc='center left', bbox_to_anchor=bbanchor)
    plt.savefig("output/images/all_uncorrectable_%s.png"%(lang), dpi=dpi)
    plt.clf()
