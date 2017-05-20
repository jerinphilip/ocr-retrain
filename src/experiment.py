from ocr import GravesOCR
import numpy as np
from error_module import Dictionary
import cv2
import sys
from aux.tokenizer import tokenize
from aux import webtotrain

from pprint import pprint
# Load preprocessing module
# Obtain (images, words)
# Obtain (sequences, targets)



book_dir='/OCRData2/minesh.mathew/Books/books_postcleaning/Malayalam/0002/'
images, truths = webtotrain.read_book(book_dir)

ocr = GravesOCR(
        "parameters/models/Malayalam.xml",  # Weights file
        "parameters/lookups/Malayalam.txt")


suggest_log_file = open("output/suggessions.log", "w+")
correct_log_file = open("output/correct.log", "w+")


error_path = 'parameters/error/Malayalam/'
kwargs = {}
for key in ['alphabet', 'save', 'words']:
    kwargs[key] = error_path + key


D = Dictionary(**kwargs)
real_word_error = 0
predicted_correct =  0
suggestion_matrix = {}
suggestion_matrix["not_found"] = 0
counter = 0
total = len(images)
for image, truth in zip(images, truths):
    counter = counter + 1
    if counter%1000 == 0:
        print("%d/%d"%(counter, total))
        print("Suggestion Matrix:\n")
        pprint(suggestion_matrix)
        print("Real word error:", real_word_error)
        print("Predicted correct:", predicted_correct)
        print("----\n", flush=True)
    predicted = ocr.recognize(image)
    if D.error(predicted) > 0:
        suggestions = D.suggest(predicted)
        print("[\t%s: [%s]\n\t%s\n]"%(predicted, ','.join(suggestions), truth), file=suggest_log_file, flush=True)
        try:
            index = suggestions.index(truth)
            if index not in suggestion_matrix:
                suggestion_matrix[index] = 0
            suggestion_matrix[index] += 1
        except ValueError:
            suggestion_matrix["not_found"] += 1

    else:
        print("[\t%s\n\t%s\n]"%(predicted, truth), file=correct_log_file, flush=True)
        if predicted != truth:
            real_word_error += 1
        else:
            predicted_correct += 1

# Feedforward into OCR, Get outputs.  
# Run postprocessing module.
# Report Errors. Some form of visualization

