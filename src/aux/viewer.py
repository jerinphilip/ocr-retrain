"""
    This script can be used to extract features from an output feature
    file of CVIT OCR Pipeline.

    Usage:
        # TODO

"""

#!/usr/bin/python3
import sys
import cv2
import numpy as np

def chunks(ls, n):
    assert(len(ls)%n == 0)
    result_ls = []
    start = 0
    while start < len(ls):
        result_ls.append(ls[start:start+n])
        start = start + n
    return result_ls

def vec2mat(ls, w,  h):
    assert (w * h == len(ls))
    return chunks(ls, w)

def extract_feature(f):
    _b, tag, truth, ftag, _e = f
    tag, embedded_feature_str = ftag.split(':')
    embedded_feature = embedded_feature_str.split(' ')
    embedded_feature.remove('')
    embedded_feature = list(map(int, embedded_feature))
    dunno1, h, w, *vectorized_image = embedded_feature
    #vectorized_image = list(map(lambda x: 255*x, vectorized_image))
    image = vec2mat(vectorized_image, w, h)
    return (np.array(image,dtype=np.float32), truth)


if __name__ == '__main__':
    with open(sys.argv[1]) as mf:
        lines = mf.read().splitlines()
        features = chunks(lines, 5)
        count = 0
        for feature in features:
            img, truth = extract_feature(feature)
            vimg = img.ravel()
            with open(str(count)+".txt", "w+") as fp:
                fp.write(' '.join(list(map(str, vimg))) + '\n')
            print(vimg, len(vimg), len(truth))
            count = count + 1

     
