from ocr import GravesOCR
import numpy as np
from error_module import Dictionary
import cv2
import sys
from aux.tokenizer import tokenize
from aux import webtotrain
from timeit import timeit
from Levenshtein import distance
from random import randint

from pprint import pprint

def stats(ocr, em, book_path):
    pagewise = webtotrain.read_book(book_dir)
    #pagewise = pagewise[-6:]
    leave_out = randint(3, len(pagewise))
    images, truths = [], []
    for imgs, ts in pagewise:
        images.extend(imgs)
        truths.extend(ts)

    em.enhance_vocabulary(truths)
    print("Recognizing..", flush=True)
    predictions = [ocr.recognize(image) for image in images]
    print("Computing Errors", flush=True)
    errors = [em.error(prediction) for prediction in predictions]
    tuples = list(zip(truths, predictions, errors))

    threshold_f = lambda x: x[2] == 0
    correct = filter(threshold_f, tuples)
    wrong = filter(lambda x: not threshold_f(x), tuples)

    stat_d = {
            "real_word_error": 0,
            "correct": 0,
            "correctable": 0,
            "incorrectable": 0
    }

    for truth, prediction, error in correct:
        if truth != prediction:
            stat_d["real_word_error"] += 1
        else:
            stat_d["correct"] += 1

    for truth, prediction, error in wrong:
        suggestions = em.suggest(prediction)
        if truth in suggestions:
            stat_d["correctable"] += 1
        else:
            stat_d["incorrectable"] += 1

    correct = sum([1 for truth,prediction in zip(truths, predictions) \
            if truth==prediction])

    stat_d["word_accuracy"] = (correct/len(images))*100
    sum_edit_distances = sum(
            [ distance(truth, prediction) \
                    for truth, prediction in zip(truths, predictions)]
            )

    total_length = sum(map(len, truths))
    stat_d["character_error_rate"] = (sum_edit_distances/total_length)*100

    stat_d["correct_without_dict"] = sum(
            [1 for truth, prediction in zip(truths, predictions)
                    if truth == prediction]
            )
    stat_d["total"] = len(images)

    return stat_d

book_dir='/OCRData2/minesh.mathew/Books/books_postcleaning/Malayalam/0002/'
#images, truths = webtotrain.read_book(book_dir)

ocr = GravesOCR(
        "parameters/models/Malayalam2.xml",  # Weights file
        "parameters/lookups/Malayalam.txt")



error_path = 'parameters/error/Malayalam/'
kwargs = {}
for key in ['alphabet', 'save', 'words']:
    kwargs[key] = error_path + key


D = Dictionary(**kwargs)
sd = stats(ocr, D, book_dir)
pprint(sd)
exit()

real_word_error = 0
predicted_correct =  0
suggestion_matrix = {}
suggestion_matrix["not_found"] = 0
counter = 0
total = len(images)



for image, truth in zip(images, truths):
    counter = counter + 1
    predicted = ocr.recognize(image)
    if D.error(predicted) > 0:
        suggestions = D.suggest(predicted)
        #print("[\t%s: [%s]\n\t%s\n]"%(predicted, ','.join(suggestions), truth), file=suggest_log_file, flush=True)
        try:
            index = suggestions.index(truth)
            if index not in suggestion_matrix:
                suggestion_matrix[index] = 0
            suggestion_matrix[index] += 1
        except ValueError:
            suggestion_matrix["not_found"] += 1

    else:
        #print("[\t%s\n\t%s\n]"%(predicted, truth), file=correct_log_file, flush=True)
        if predicted != truth:
            real_word_error += 1
        else:
            predicted_correct += 1
    if counter%1000 == 0 or counter == total: 
        print("%d/%d"%(counter, total))
        print("Suggestion Matrix:\n")
        pprint(suggestion_matrix)
        print("Real word error:", real_word_error)
        print("Predicted correct:", predicted_correct)
        print("----\n", flush=True)

# Feedforward into OCR, Get outputs.  
# Run postprocessing module.
# Report Errors. Some form of visualization

