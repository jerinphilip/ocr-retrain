from .params import params
from collections import Counter
from copy import deepcopy
import pdb
import sys
from doctools.scripts.debug import time

#@time
def naive(predictions, truths, dictionary):
    # Count number of errored words
    cost, errors = 0, 0
    for prediction, truth in zip(predictions, truths):
        if dictionary.error(prediction):
            # Language model sees as error.
            if prediction != truth:
                cost += params["type"]
            else:
                cost += params["verify"]
        else:
            cost += params["ignore"]
            if prediction != truth:
                errors += 1


    # Full typing cost
    return (cost, errors)
#@time
def suggest(predictions, truths, dictionary):
    cost, errors = 0, 0
    for prediction, truth in zip(predictions, truths): 
        if dictionary.error(prediction):
            if prediction != truth:
                suggestions = dictionary.suggest(prediction)
                if truth in suggestions:
                    cost += params["dropdown"]
                else:
                    cost += params["type"]
            else:
                cost += params["verify"]
        else:
            cost += params["ignore"]
            if prediction != truth:
                errors += 1
    
    return (cost, errors)

#@time
def cluster(predictions, truths, dictionary, components):
    def compute(component):
        # Obtain errored indices

        def get(array, indices):
            return [array[i] for i in indices]

        def detect(cs):
            _errors = 0
            errored_indices = []
            for i in cs:
                prediction, truth = predictions[i], truths[i]
                if dictionary.error(prediction):
                    errored_indices.append(i)
                else:
                    if prediction != truth:
                        _errors += 1
            return (set(errored_indices), _errors)

        def check(i):
            correct = (truths[i] == predictions[i])
            eq_select = (selection == predictions[i])
            return (correct and eq_select)

        def select(remaining, selection):
            icheck = lambda i: not check(i)
            selected = set(filter(check, remaining))
            return selected

        def candidate(pool):
            if not pool:
                return (None, 0)
            component_predictions = get(predictions, pool) 
            predictions_counter = Counter(component_predictions)
            best = max(predictions_counter.items(), key=lambda x: x[1])
            selection, count = best
            return selection, count

        def fpool(pool, selection, selected):
            for i, pred in enumerate(predictions):
                if pred == selection and i in pool:
                    pool.remove(i)
            pool = pool - selected
            return pool

        def batch_correctable(errored, pool):
            selection, count = candidate(pool)
            if count <= 1:
                return False
            
            # At least one prediction needs to be equal to selection.
            for i in errored:
                if selection == predictions[i]:
                    return True

            return False



        ccost, cerrors = 0, 0
        errored, rwe = detect(component)
        cerrors += rwe
        pool = set(deepcopy(component))

        selected, deselected = set(), set()
        while True:
            selection, count = candidate(pool)
            selected = select(errored, selection)
            deselected = errored - selected
            pool = fpool(pool, selection, selected)
            
            #ccost += params["deselection"] * len(deselected) + \
                    #params["selection"] * len(selected)
            ccost += params["selection"]*len(selected)

            errored = deselected
            #assert(errored.intersection(pool) == errored)
            if not batch_correctable(errored, pool):
                break

        subpreds = get(predictions, deselected)
        subtruths = get(truths, deselected)
        _cost, _errors = suggest(subpreds, subtruths,
            dictionary)
        ccost += _cost
        cerrors += _errors
        complete = True

        return (ccost, cerrors)
    
    fcost,ferrors = 0, 0
    for component in components:
        _cost, _errors, = compute(component)
        fcost += _cost
        ferrors += _errors

    #print(n_added_in, n_left_out, file=sys.stderr)
    return (fcost, ferrors)
