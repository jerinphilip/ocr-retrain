import json
import os
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

def generate_required(save):
    book_ps = save["book_dir"].split('/')
    book = book_ps[-2]
    keys = map(lambda x: x/10, range(10))
    skeys = map(str, keys)
    xs, cp, cbp, icbp = [], [], [], []
    for known_percent in skeys:
        unseen = save[known_percent]["unseen"]
        correct_percent = unseen["correct"]/unseen["total"]

        to_suggest = unseen["total"] - \
                (unseen["correct"] + unseen["real_word_error"])

        correctable_percent = unseen["correctable"]/to_suggest
        uncorrectable_percent = unseen["uncorrectable"]/to_suggest
        xs.append(known_percent)
        cp.append(correct_percent)
        cbp.append(correctable_percent)
        icbp.append(uncorrectable_percent)

    return (xs, cp, cbp, icbp, book)


hi_fmap = dict(map(lambda x: x..strip().split('_'), open("hi.fmap")))
ml_fmap = dict(map(lambda x: x..strip().split('_'), open("ml.fmap")))

saves = []

for dr, drs, fls in os.walk('output/hi-ocr'):
    for fn in fls:
        fn_with_path = dr + '/' + fn
        saves.append(json.load(open(fn_with_path)))

values = map(generate_required, saves)
xs, cps, cbps, icbps, books = zip(*values)
for x, y, book in zip(xs, cps, books):
    plt.plot(x, y)
    plt.xlabel("fraction of data in vocabulary")
    plt.ylabel("correct percent")

plt.savefig("output/images/all_correct_hi.png", dpi=200)
plt.clf()

for x, y, book in zip(xs, cbps, books):
    plt.plot(x, y)
    plt.xlabel("fraction of data in vocabulary")
    plt.ylabel("correctable percent")

plt.savefig("output/images/all_correctable_hi.png", dpi=200)
plt.clf()

for x, y, book in zip(xs, icbps, books):
    plt.plot(x, y)
    plt.xlabel("fraction of data in vocabulary")
    plt.ylabel("uncorrectable percent")

plt.savefig("output/images/all_uncorrectable_hi.png", dpi=200)
plt.clf()
