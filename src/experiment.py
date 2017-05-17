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



book_dir='/OCRData2/minesh.mathew/Books/books_postcleaning/Malayalam/0002/'
images, truths = webtotrain.read_book(book_dir)

ocr = GravesOCR(
        "models/Malayalam.xml",  # Weights file
        "lookups/Malayalam.txt")


suggest_log_file = open("suggessions.log", "w+")
correct_log_file = open("correct.log", "w+")

D = Dictionary(lang="malayalam")
real_word_error = 0
predicted_correct =  0
suggestion_matrix = {}
suggestion_matrix["not_found"] = 0
counter = 0
total = len(images)
for image, truth in zip(images, truths):
    counter = counter + 1
    print("%d/%d"%(counter, total))
    predicted = ocr.recognize(image)
    if D.error(predicted) > 0:
        suggestions = D.suggest(predicted)
        print("[\t%s: [%s]\n\t%s\n]"%(predicted, ','.join(suggestions), truth), file=suggest_log_file)
        try:
            index = suggestions.index(truth)
            if index not in suggestion_matrix:
                suggestion_matrix[index] = 0
            suggestion_matrix[index] += 1
        except ValueError:
            suggestion_matrix["not_found"] += 1

    else:
        print("[\t%s\n\t%s\n]"%(predicted, truth), file=correct_log_file)
        if predicted != truth:
            real_word_error += 1
        else:
            predicted_correct += 1


pprint(suggestion_matrix)
pprint(real_word_error)
# Feedforward into OCR, Get outputs.  
# Run postprocessing module.
# Report Errors. Some form of visualization

