
from doctools.configs.params import params
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
    # Check cluster purity.

    def get(array, indices):
        return [array[i] for i in indices]

    def compute(component):
        incorrect = [i for i in component if dictionary.error(predictions[i])]
        correct  = [i for i in component if not dictionary.error(predictions[i])]
        cpreds = get(predictions, incorrect)

        candidate, frequency = None, 0
        if incorrect:
            candidate, frequency = max(Counter(cpreds).items(), 
                    key=lambda x: x[1])
        else:
            candidate, frequency = None, 0

        pure = [i for i in incorrect if candidate == truths[i]]
        impure = [i for i in incorrect if candidate != truths[i]]


        #print("pure", len(pure), "impure", len(impure), "incorrects", len(incorrect))
        #input()

        cost = params["verify"]*(bool(pure))

        spreds = get(predictions, impure)
        struths = get(truths, impure)
        _cost, _errors = suggest(spreds, struths, dictionary)

        rwes = sum([1 for i in correct if predictions[i] != truths[i]])
        return (cost + _cost, _errors + rwes)

    ls = map(compute, components)
    costs, errors = list(zip(*ls))

    return (sum(costs), sum(errors))

