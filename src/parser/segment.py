"""
This file has auxilliary functions to handle jLayout's segment output.

"""

import cv2

def segment(img) 
    # This can fail. 
    # Wrap around an error handler, maybe?
    def crop(bbox):
        x, y, w, h = bbox
        return img[y:y+h, x:x+w]
    return crop

def read(tiff, **kwargs):
    """
    Expects jLayout to be already run.
    tiff.{lines,blocks,words}.txt should be present.
    """
    if 'unit' not in kwargs: kwargs['unit'] = 'line'
    fname = tiff + '.%ss.txt'%(kwargs['unit'])
    extract = lambda x: list(map(int, x.strip().split()))
    bboxes = map(extract, fp)
    img = cv2.imread(tiff)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(grayscale, 0, 1, cv2.THRESH_OTSU)

    segments = list(map(segment(binarized), bboxes))

    # Include a routine to sort by bboxes if necessary here.
    # Generate a 2D array of segments
    # Which can be later joined together

    return segments

