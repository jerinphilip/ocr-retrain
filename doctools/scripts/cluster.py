from doctools.cluster.mst import cluster, merge
from doctools.cluster.distance import jaccard, lev, euc, cos
from doctools.parser.convert import page_to_unit
from doctools.meta.file_locs import get_pickeled, get_clusters
from doctools.parser import webtotrain
from argparse import ArgumentParser
from pprint import pprint
from .dot import as_dot
# from distance import levenshtein
import json
from functools import partial
import pdb
from doctools.ocr import GravesOCR
import os
from .opts import base_opts
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import cv2
import pdb
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from .debug import time
import pickle
def get_index(a):
    uniques = np.unique(np.array(a))
    idx = {u:[] for u in uniques}
    for i, v in enumerate(a):
        idx[v].append(i)
    return dict(sorted(idx.items(),key=lambda x: -len(x[1])))
def save(**kwargs):
    data = kwargs['data']
    meta = kwargs['book']
    loc = kwargs['outpath']
    if not kwargs['feat']:
        with open('%s/%s.pkl'%(loc, meta),'wb') as f:
            pickle.dump(data, f)
        print('Book predictions saved..')
    else:
        with open('%s/%s_features_cluster.pkl'%(loc, meta),'wb') as f:
            pickle.dump(data, f)
        print('Book featutres saved..')

def visualize(path, components):
    with open(os.path.join(fpath, 'annotation.txt'), 'r') as in_file:
        lines = in_file.readlines()
    words = [line.rsplit()[1] for line in lines]
    number_of_subplots = 10
    images = [line.rsplit()[0] for line in lines]
    for i,v in enumerate(components):
        print("cluster no: %d"%i)
        number_of_subplots = min(len(v), number_of_subplots)
        for j in range(number_of_subplots):
            
            print(os.path.join(path,images[v[j]]))
            im = cv2.imread(os.path.join(path,images[v[j]]))
            print("subploting [%d/%d]"%(j,len(v)))
            print(1,number_of_subplots, j+1)
            plt.subplot(1,number_of_subplots, j+1),plt.xticks([]), plt.yticks([])
            plt.imshow(im)
        plt.savefig(os.path.join(args.output,'cluster_%d.png'%i))
    # plt.show() 


def k_nearest(fpath):
    words=[]
    with open(os.path.join(fpath, 'annotation.txt'), 'r') as in_file:
        lines = in_file.readlines()
    words = [line.rsplit()[1] for line in lines]
    neigh = KNeighborsClassifier(n_neighbors=3) 
    test_words = np.array(words[2000:])
    test_features = np.array(features[2000:])
    test_index = get_index(test_words)
    indices = sum([v  for i,v in test_index.items() if len(v)>=3],[])
    test_words = test_words[indices]
    test_features = test_features[indices]
    neigh.fit(features[:2000], words[:2000])
    
    y_predict = neigh.predict(test_features)
    pdb.set_trace()
    acc = [1 if y_predict[i]==test_words[i] else 0 for i in  range(len(test_words))]
    print(sum(acc)/len(test_words))
@time
def get_predictions(fpath):
    print("Reading book...", end='', flush=True)
    pagewise = webtotrain.read_book(fpath)
    
    print("Done")
    images, truths = page_to_unit(pagewise)
    print("Predicting....", end='', flush=True)
    predictions = ocr.predict(images)
    print("Done")
    errored = [predictions[i] for i in range(len(truths)) if predictions[i] != truths[i]]
    return errored, predictions

def get_images(fpath):
    print("Reading book...", end='', flush=True)
    pagewise = webtotrain.read_book(fpath)
    print("Done")
    images, truths = page_to_unit(pagewise)
    return images, truths
@time
def get_features(fpath):
    print('Loading features...')
    feat = np.load(os.path.join(fpath, "feats.npy"))
    features = [feat[i] for i in range(feat.shape[0])]
    print('Done....')
    return features
@time
def form_clusters(elements, **kwargs):
    dist  = kwargs["distance"]
    print("Clustering....", end='', flush=True)
    edges, components = cluster(elements, dist, threshold=0.5, prune_above=0.8, rep='components')
    print("Done")
    return edges, components
@time
def find_examplars(all_edges):
    return set([ex for ex, c in all_edges])
@time
def group_components(exemplars, all_edges):
    all_comp = []
    for e in exemplars:
        comp = []
        for ex, cl in all_edges:
            if ex == e:
                comp.append(cl)
        all_comp.append(comp)
    return all_comp


if __name__ == '__main__':
 
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()
    config_file = open(args.config)
    config = json.load(config_file)

    # Load OCR
    print(config["model"])
    ocr = GravesOCR(config["model"], config["lookup"])
    book_list =['0022', '0029', '0040', '0060', '0061','0191', '0069', '0211']
    # book_list = ['0191']
    # Parse Book in and predict
    book_name = book_list[args.book]
    outpath = args.output
    outpath_pickled = os.path.join(args.output, 'pickled')
    # # neigh = KNeighborsClassifier(n_neighbors=3)
    features = get_features(os.path.join(config["feat_dir"], book_name))
    print(book_name)
    images, truths = get_images(os.path.join(config["dir"], book_name))
    # k_nearest(os.path.join(config["feat_dir"], book_name))
    if len(images) == len(features):

        predictions = get_pickeled(book_name, type="predictions")
        # edges_word, comp_words = form_clusters(predictions, lev)
        print("Into features now ..... :/ ")
        if get_pickeled(book_name, type="edges")!= None:
            edges_feat = get_pickeled(book_name, type="edges")
            
        else:
            edges_feat, comp_feat = form_clusters(features, distance= euc)
            save(data = edges_feat, book=book_name, outpath=outpath_pickled, feat=True)


        pdb.set_trace()

        
    #     edges_combined, components_combined = merge(edges_word, edges_feat, predictions)

    #     exemplars = find_examplars(edges_combined)
        
    #     mydict = dict(zip(list(exemplars), components_combined))
    #     with open('%s/jsons/%s.json'%(outpath, book_name), 'w+') as fp:
    #            json.dump(mydict, fp, indent=4)
    # else:
    #     print('for book %s features and images did not match'%book_name)


    print("Finished...")

    

    # for c, component in enumerate(comp_words):
    #     #rep = as_dot(predictions, edges, component) 
    #     rep = as_dot(errors, edges_word, comp_word) 
    #     of = os.path.join(args.output, "{}.dot".format(c))
    #     #of = "{}_{}.dot".format(args.output, c)
    #     with open(of, "w+") as ofp:
    #         ofp.write(rep)
    
    # visualize(fpath, components)
    