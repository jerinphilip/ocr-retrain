
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
from aux import webtotrain
from timeit import timeit
from Levenshtein import distance
from random import randint
import json
from pprint import pprint
from functools import partial

def stats(ocr, em, book_path):
    pagewise = webtotrain.read_book(book_path)
    #pagewise = pagewise[-6:]
    leave_out = randint(3, len(pagewise))
    images, truths = [], []
    for imgs, ts in pagewise:
        images.extend(imgs)
        truths.extend(ts)

    #em.enhance_vocabulary(truths)
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



