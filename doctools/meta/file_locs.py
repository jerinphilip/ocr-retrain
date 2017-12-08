import pickle
import json
import os
import pdb

def outdir(*names):
    base_dir = '/data5/deepayan/new_ocr/doctools/outdir/'
    return os.path.join(base_dir, *names)

path = {
    "pickle": {
            "predictions": outdir('pickled'),
            "features": outdir('pickled')
    },
    "json":{
            "words": outdir('jsons', 'feat'),
            "images": outdir('jsons', 'images'),
            "combined": outdir('jsons', 'combined'),
            "cost": outdir('jsons', 'cost') 
    }
}

def load(book_name, **kwargs):
    base_dir = outdir('pickled')
    fname = "%s.%s.pkl"%(book_name, kwargs["feat"])
    fpath = os.path.join(base_dir, fname)
    if os.path.exists(fpath):
        with open(fpath, 'rb') as fp:
            saved = pickle.load(fp)
            return saved
    else:
        return None
        

def get_pickeled(book_name, **kwargs):
    flag = kwargs["type"]
    edges_feat, predictions = None, None
    if flag == "predictions":
        if os.path.exists(os.path.join(path["pickle"]["predictions"],'%s.pkl'%book_name)):
            print('Loading predictions...')
            with open(os.path.join(path["pickle"]["predictions"],'%s.pkl'%book_name), 'rb') as f:
                predictions = pickle.load(f)
            return(predictions)
        
    if flag == "edges":
        if os.path.exists(os.path.join(path["pickle"]["features"],'%s.features_cluster.pkl'%book_name)):
            print('Loading Edges...')
            with open(os.path.join(path["pickle"]["features"],'%s.features_cluster.pkl'%book_name), 'rb') as f:
                edges_feat = pickle.load(f)
        return(edges_feat)
    if flag == "word edges":
        if os.path.exists(os.path.join(path["pickle"],'%s.words_cluster.pkl'%book_name)):
            print('Loading Edges...')
            with open(os.path.join(path["pickle"],'%s.words_cluster.pkl'%book_name), 'rb') as f:
                edges_feat = pickle.load(f)
        return(edges_feat)
        

def get_clusters(book_name, **kwargs):
    features = kwargs["features"]
    data=None
    # pdb.set_trace()
    if os.path.exists(os.path.join(path["json"]["%s"%features], '%s.json'%book_name)):
        with open(os.path.join(path["json"]["%s"%features], '%s.json'%book_name)) as json_data:
            data = json.load(json_data)
    return data
    
        

