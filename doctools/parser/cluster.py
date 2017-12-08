import pickle
import json
import os
import pdb
from pprint import pprint

def outdir(*names):
    base_dir = '/OCRData2/saves'
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

def dict_equal(first, second):
    pprint(first)
    pprint(second)
    for key in first:
        if key not in second: return False
        elif first[key] != second[key]: return False
    return True

def load(book_name, **kwargs):
    base_dir = outdir('pickled')
    fname = "%s.%s.pkl"%(book_name, kwargs["feat"])
    fpath = os.path.join(base_dir, fname)
    params_path = fpath + '.params'
    if os.path.exists(fpath) and os.path.exists(params_path):
        with open(params_path, 'rb') as pfp:
            params = pickle.load(pfp)
            with open(fpath, 'rb') as fp:
                saved = pickle.load(fp)
                changed = not dict_equal(params, kwargs)
                return (saved, changed)
    else:
        return (None, False)

def gmkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save(**kwargs):
    base_dir = outdir('pickled')
    data = kwargs['data']
    meta = kwargs['book']
    gmkdir(base_dir)
    fname = "%s.%s.pkl"%(meta, kwargs["feat"])
    fpath = os.path.join(base_dir, fname)
    params_path = fpath + '.params'
    with open(fpath, "wb+") as f:
        pickle.dump(data, f)
        print("Saving:", kwargs["feat"])

    with open(params_path, "wb+") as f:
        pickle.dump(kwargs["params"], f)
