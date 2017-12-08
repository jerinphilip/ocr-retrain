from doctools.cluster.mst import cluster, merge
from doctools.cluster.distance import jaccard, lev, euc, cos
from doctools.parser.convert import page_to_unit
from doctools.meta.file_locs import get_pickeled, get_clusters
from doctools.cluster.k_nearest.distance import normalized_euclid_norm
from doctools.parser import read_book
from argparse import ArgumentParser
from pprint import pprint
# from .dot import as_dot
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
    if kwargs['feat'] == "predictions":
        with open('%s/%s.pkl'%(loc, meta),'wb') as f:
            pickle.dump(data, f)
        print('Book predictions saved..')
    if kwargs['feat'] == "images":
        with open('%s/%s.features_cluster.pkl'%(loc, meta),'wb') as f:
            pickle.dump(data, f)
        print('image featutres saved..')
    if kwargs['feat'] == "words":
        with open('%s/%s.words_cluster.pkl'%(loc, meta),'wb') as f:
            pickle.dump(data, f)
        print('words featutres saved..')    

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
    pagewise = read_book(book_path=fpath, unit='word')
    
    print("Done")
    images, truths = page_to_unit(pagewise)
    print("Predicting....", end='', flush=True)
    predictions = ocr.predict(images)
    print("Done")   
    errored = [predictions[i] for i in range(len(truths)) if predictions[i] != truths[i]]
    return errored, predictions

def get_images(fpath):
    print("Reading book...", end='', flush=True)
    pagewise = read_book(book_path=fpath, unit='word')
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
    threshold = kwargs["threshold"]
    print("Clustering....", end='', flush=True)
    edges, components = cluster(elements, dist, threshold=threshold,prune_above=0.8, rep='components')
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
    # book_list =['0191', '0029', '0040', '0060', '0061', '0069', '0211']
    # book_list = ['0191']
    # Parse Book in and predict
    book_name = config["books"][args.book]
    outpath = args.output
    outpath_pickled = os.path.join(args.output, 'pickled')
    # # neigh = KNeighborsClassifier(n_neighbors=3)
    features = get_features(os.path.join(config["feat_dir"], book_name))
    print(book_name)
    images, truths = get_images(os.path.join(config["dir"], book_name))
    # k_nearest(os.path.join(config["feat_dir"], book_name))
    if len(images) == len(features):

        if get_pickeled(book_name, type="predictions")!=None:
            predictions = get_pickeled(book_name, type="predictions")
            edges_word, comp_words = form_clusters(predictions, distance=lev, threshold=0.5)
            save(data = {"components": comp_words, "edges": edges_word},book=book_name, feat="words", outpath=outpath_pickled)
        else:
            print("predicting....")
            predictions = ocr.predict(images)
            edges_word, comp_words = form_clusters(predictions, distance=lev, threshold=0.5)
            save(data = {"components": comp_words, "edges": edges_word},book=book_name, feat="words", outpath=outpath_pickled)
            save(data=predictions, book=book_name, feat= "predictions", outpath=outpath_pickled)
        print("Into features now ..... :/ ")
        if get_pickeled(book_name, type="edges")!= None:
            # data = get_pickeled(book_name, type="edges")
            # edges_feat, comp_feat = data["edges"], data["components"]
             edges_feat, comp_feat = form_clusters(features, distance= normalized_euclid_norm, threshold=0.36)
             save(data = {"components":comp_feat,"edges":edges_feat}, book=book_name, outpath=outpath_pickled, feat="images")

        else:
            edges_feat, comp_feat = form_clusters(features, distance= normalized_euclid_norm, threshold=0.36)
            save(data = {"components":comp_feat,"edges":edges_feat}, book=book_name, outpath=outpath_pickled, feat="images")

        
        edges_combined, components_combined = merge(edges_word, edges_feat, predictions)
        
        # exemplars = find_examplars(edges_feat)
        
        # mydict = dict(zip(list(exemplars), edges_feat))
        with open('%s/jsons_word/%s.json'%(outpath, book_name), 'w+') as fp:
               json.dump(comp_words, fp, indent=4)
        with open('%s/jsons_feat/%s.json'%(outpath, book_name), 'w+') as fp:
               json.dump(comp_feat, fp, indent=4)
        with open('%s/jsons/%s.json'%(outpath, book_name), 'w+') as fp:
               json.dump(components_combined, fp, indent=4)
    else:
        print('for book %s features and images did not match'%book_name)


    print("Finished...")

    

    # for c, component in enumerate(comp_words):
    #     #rep = as_dot(predictions, edges, component) 
    #     rep = as_dot(errors, edges_word, comp_word) 
    #     of = os.path.join(args.output, "{}.dot".format(c))
    #     #of = "{}_{}.dot".format(args.output, c)
    #     with open(of, "w+") as ofp:
    #         ofp.write(rep)
    
    # visualize(fpath, components)
    