from .params import params
from collections import Counter
from copy import deepcopy
import pdb
from doctools.scripts.debug import time

@time
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
@time
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

@time
def cluster(predictions, truths, dictionary, components):
    cost, errors = 0, 0

    # Group predictions by components.
    # Assume all gets corrected in one go.

    def get(array, indices):
        return [array[i] for i in indices]

    def get_indices(array, item):
        indices=[]
        for i, v in enumerate(array):
            if item ==  v:
                indices.append(i)
        return indices

    def compute(component):

        # Obtain errored indices
        errored_indices = []
        cost, errors = 0, 0
        for i in component:
            # pdb.set_trace
            prediction, truth = predictions[int(i)], truths[int(i)]
            if dictionary.error(prediction):
                errored_indices.append(int(i))
            else:
                if prediction != truth:
                    errors += 1

        # Generate suggestions from component truths
        

        indices = deepcopy(component)
        complete = False
        while not complete:
            # From remaining indices, choose most frequent as the
            # correct answer.
            component_predictions = get(predictions, indices) 
            predictions_counter = Counter(component_predictions)
            best = max(predictions_counter.items(), key=lambda x: x[1])
            selection, count = best
            
            # If they are all different, serves no purpose at all as
            # well, no batch correction.
            if count > 1:
                # Find what's not equal to selection, leave it out.
                left_out = []
                added_in = []

                for i in errored_indices:
                    if predictions[i] != selection or \
                            predictions[i] != truths[i]:
                        left_out.append(i)
                    else:
                        added_in.append(i)


                # Whatever's left out is errored_indices now
                errored_indices = left_out

                # Remove corrected from the component
                for i in indices:
                    if predictions[i] == selection:
                        indices.remove(i)


                # Each batch correction costs this much from an annotator.
                # This can be made proportional to the entries
                # cost += params["cluster"]
                cost += params["deselection"] * len(left_out) + \
                        params["selection"] * len(added_in)

                # How do we know when the process completes.
                # Whatever is left out, is it correct at all, to choose a
                # best?
                ps = []
                for i in left_out:
                    possible = (truths[i] == predictions[i])
                    ps.append(possible)

                # If any of ps is true, possibility of correction.
                complete = not any(ps)

            else:
                subpreds = get(predictions, errored_indices)
                subtruths = get(truths, errored_indices)
                _cost, _errors = suggest(subpreds, subtruths,
                    dictionary)
                cost += _cost
                errors += _errors
                complete = True

        return (cost, errors)
    
    cost, errors = 0, 0
    for component in components:
        _cost, _errors = compute(component)
        cost += _cost
        errors += _errors

    return (cost, errors)
