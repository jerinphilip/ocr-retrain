from ocr import GravesOCR
import numpy as np
from error_module import Dictionary
import cv2
import sys
from aux.tokenizer import tokenize
from aux import webtotrain

# Load preprocessing module
# Obtain (images, words)
# Obtain (sequences, targets)


book_dir = '/home/jerin/honours-super/data/0002/'
images, truths = webtotrain.read_book(book_dir)

ocr = GravesOCR(
        "models/Malayalam.xml",  # Weights file
        "lookups/Malayalam.txt")

D = Dictionary(lang="malayalam")

real_word_error = 0
suggestion_matrix = {}
for image, truth in zip(images, truths):
    predicted = ocr.recognize(image)
    print("[\t%s\n\t %s\n]"%(predicted, truth))
    if D.error(predicted) > 0:
        suggestions = D.suggest(predicted)
        index = suggestions.index(truth)
        if index not in suggestion_matrix:
            suggestion_matrix[index] = 0
        suggestion_matrix[index] += 1

    else:
        if predicted != truth:
            real_word_error += 1


pprint(suggestion_matrix)
pprint(real_word_error)
# Feedforward into OCR, Get outputs.  
# Run postprocessing module.
# Report Errors. Some form of visualization

