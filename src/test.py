from ocr import GravesOCR
import numpy as np
from error_module import Dictionary
import cv2
import sys
from aux.tokenizer import tokenize

def convert(pyocr_output):
    codepoint = pyocr_output[1:]
    codepoint_value = int(codepoint, 16)
    return chr(codepoint_value)

def display_img(inputs):
    shape = (32, len(inputs)//32)
    shape = tuple(reversed(shape))
    img = 255*np.array(inputs, dtype=np.uint8).reshape(shape).T
    cv2.imshow("img", img)
    cv2.waitKey(0)

ocr = GravesOCR(
        "models/Malayalam.xml",  # Weights file
        "lookups/Malayalam.txt")

fname = sys.argv[1]
img = cv2.imread(fname)

with open(fname + '.lines.txt') as fp:
#with open(fname + '.words.txt') as fp:
    bboxes = map(lambda x: list(map(int, x.strip().split())), fp)
    for x, y, w, h in bboxes:
        x2, y2 = x+w, y+h
        cropped_img = img[y:y+h, x:x+w]
        grayscale = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        ret2,bw = cv2.threshold(grayscale,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        height, width = bw.shape
        f = 32/height
        bw_32 = cv2.resize(bw, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        #print(bw_32.shape)

        #cv2.imshow("window", 255*bw_32)
        #cv2.waitKey(0)
        H, W = bw_32.shape
        vector = []
        for j in range(W):
            for i in range(H):
                vector.append(float(bw_32[i][j]))
        vector = list(map(float,bw_32.T.ravel()))

        #print(type(vector))
        #print(vector)
        s = ocr.recognize(vector)
        for w in tokenize(s):
            print(w)


        #topleft = (min(x2, x), min(y2, y))
        #bottomright = (max(x2, x), max(y2, y))
        #print(x, y, w, h)
        #cv2.rectangle(img,topleft, bottomright, (0,255,0),3)

#cv2.imwrite("output.jpg", img)



