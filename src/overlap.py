import sys 
from aux import webtotrain
from error_module import Dictionary
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted


# Global one time load of dictionary
error_path = 'parameters/error/Malayalam/'
kwargs = {}
for key in ['alphabet', 'save', 'words']:
    kwargs[key] = error_path + key

D = Dictionary(**kwargs)

keys = D.trie.keys()
unique_keys = set(keys)

def get_metrics(book):
    book_dir='/OCRData2/minesh.mathew/Books/books_postcleaning/Malayalam/'+book+'/'
    pagewise = webtotrain.read_book(book_dir)
    images, truths = [], []
    for imgs, ts in pagewise:
        truths.extend(ts)

    # We now have truths in truths
    unique_truths = set(truths)
    extra_vals = unique_truths
    values = compute_metrics(unique_truths)

    return (extra_vals, values)

def compute_metrics(unique_truths):
    keys_and_truths = unique_truths & unique_keys
    n_keys_union_truths = len(keys_and_truths)


    n_keys_alone = len(unique_keys) - n_keys_union_truths
    n_truths_alone = len(unique_truths) - n_keys_union_truths



    values = [
            ("Truths", n_truths_alone),
            ("Vocabulary", n_keys_alone),
            ("TUV", n_keys_union_truths)
    ]

    return values


def plot_venn(subsets, labels):
    # Subset sizes

    v = venn2_unweighted(subsets=subsets, set_labels=labels)
    values = subsets
    s = sum(values)
    vals = tuple(map(lambda x: round((x/s)*100, 2), values))

    strs = list(map(lambda t, s: "%.2f%%\n%d"%(s,t), subsets, vals))

    # Subset labels
    v.get_label_by_id('10').set_text(strs[0])
    v.get_label_by_id('01').set_text(strs[1])
    v.get_label_by_id('11').set_text(strs[2])

    # Subset colors
    v.get_patch_by_id('10').set_color('c')
    v.get_patch_by_id('01').set_color('#993333')
    v.get_patch_by_id('11').set_color('blue')

    # Subset alphas
    v.get_patch_by_id('10').set_alpha(0.4)
    v.get_patch_by_id('01').set_alpha(1.0)
    v.get_patch_by_id('11').set_alpha(0.7)

    # Border styles
    #c = venn2_circles(subsets=subsets, linestyle='solid')
    #c[0].set_ls('dashed')  # Line style
    #c[0].set_lw(2.0)       # Line width

    plt.show()

book_index = {}

with open("fmap.list") as mp:
    book_index = dict(map(lambda x: x.split('_'), mp.read().splitlines()))


with open("ml.list") as f:
     books = f.read().splitlines()
     #books = books[4:5]
     results = []

     unique_truths = set()
     for book in books:
         ut, v = get_metrics(book)
         unique_truths = unique_truths.union(ut)
         results.append(v)

     size = 5
     plt.figure(figsize=(size, size))
     header = None
     headers = None
     out = []
     vals_ls = [0 for i in range(3)]
     for i in range(len(books)):
         headers, values = zip(*results[i])
         header = ','.join(headers)
         hit_percent = round((values[2]/(values[0] + values[2]))*100, 2)
         strs = [books[i], book_index[books[i]], str(hit_percent)] + list(map(str, values))
         out.append(','.join(strs))


     """

     """

     with open("stats.csv", "w+") as op:
         print(header, file=op)
         print('\n'.join(out), file=op)
        
     headers, values = zip(*compute_metrics(unique_truths))
     print(headers, values)
     print('\t'.join(map(str, values)))
     print(headers, values)
     print(len(unique_keys), len(unique_truths))
     plot_venn(values, headers)
     plt.title("Overall Books")
     plt.savefig('images/file.png')
     plt.clf()





