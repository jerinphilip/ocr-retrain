
#!/usr/bin/python3

#SBATCH --output=output/parallel-%j.out
#SBATCH --partition=atom-cpu
#SBATCH -n 16 # 10 cores
#SBATCH -N 7
#SBATCH --overcommit

import sys
import os
import multiprocessing

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
sys.path.append(os.getcwd())


from ocr import GravesOCR
import numpy as np
from error_module import Dictionary
import cv2
import sys
from aux.tokenizer import tokenize
from parser import webtotrain
from timeit import timeit
from Levenshtein import distance
from random import randint
import json
from pprint import pprint
from functools import partial

def split(data, fraction):
    assert(0 <= fraction && fraction <= 1)
    total = len(data)
    num_first = floor(fraction*total)
    num_second = total - num_first
    first = data[:num_first]
    second = data[num_first:]
    return (first, second)

def stats(ocr, em, book_path):
    pagewise = webtotrain.read_book(book_path)
    images, truths = [], []
    for imgs, ts in pagewise:
        images.extend(imgs)
        truths.extend(ts)

    print("Recognizing..", flush=True)
    predictions = [ocr.recognize(image) for image in images]

    stat_d = {}

    for fraction in [0.2, 0.4, 0.6, 0.8]:
        first, second = split(pagewise, fraction)
        vocab_from_book = []
        for imgs, ts in first:
            vocab_from_book.extend(ts)
        em.enhance_vocabulary(vocab_from_book)
        print("Computing Errors", flush=True)
        errors = [em.error(prediction) for prediction in predictions]
        tuples = list(zip(truths, predictions, errors))

        threshold_f = lambda x: x[2] == 0
        correct = filter(threshold_f, tuples)
        wrong = filter(lambda x: not threshold_f(x), tuples)

        sfd = {
                "real_word_error": 0,
                "correct": 0,
                "correctable": 0,
                "incorrectable": 0
        }

        for truth, prediction, error in correct:
            if truth != prediction:
                sfd["real_word_error"] += 1
            else:
                sfd["correct"] += 1

        for truth, prediction, error in wrong:
            suggestions = em.suggest(prediction)
            if truth in suggestions:
                sfd["correctable"] += 1
            else:
                sfd["incorrectable"] += 1

        correct = sum([1 for truth,prediction in zip(truths, predictions) \
                if truth==prediction])

        sfd["word_accuracy"] = (correct/len(images))*100
        sum_edit_distances = sum(
                [ distance(truth, prediction) \
                        for truth, prediction in zip(truths, predictions)]
                )

        total_length = sum(map(len, truths))
        sfd["character_error_rate"] = (sum_edit_distances/total_length)*100
        sfd["correct_without_dict"] = sum(
                [1 for truth, prediction in zip(truths, predictions)
                        if truth == prediction]
                )
        sfd["total"] = len(images)
        stat_d[fraction] = sfd

    stat_d["book_dir"] = book_path
    return stat_d

def work(config, book_loc):
    ocr = GravesOCR(config["model"], config["lookup"])
    error = Dictionary(**config["error"])
    statd = stats(ocr, error, book_loc)
    return statd


if __name__ == '__main__':
    config = json.load(open(sys.argv[1]))

    ncpus = None
    # get number of cpus available to job
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = multiprocessing.cpu_count()

    print("ncpus found:", ncpus)
    partial_work = partial(work, config)
    book_locs = map(lambda x: config["dir"] + x + '/', config["books"])
    p = multiprocessing.Pool(ncpus)
    stats_ls = p.map(partial_work, book_locs)
    p.close()
    p.join()
    print(list(stats_ls))



