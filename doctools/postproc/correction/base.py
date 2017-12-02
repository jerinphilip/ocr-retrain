from .params import params
from collections import Counter

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

def cluster(predictions, truths, dictionary, components):
    cost, errors = 0, 0

    # Group predictions by components.
    # Assume all gets corrected in one go.

    def get(array, indices):
        return [array[i] for i in indices]
    
    def compute(component):
        # Obtain errored indices
        errored_indices = []
        for i in components:
            prediction, truth = predictions[i], truths[i]
            if dictionary.error(prediction):
                errored_indices.append(i)
            else:
                if prediction != truth:
                    errors += 1

        # Generate suggestions from component truths
        component_truths = get(truths, component) 
        truths_counter = Counter(component_truths)
        best = max(truths_counter.items(), key=lambda x: x[1])
        selection, _ = best
        cost, errors = 0, 0

        # TODO enhance this section with more complexity
        for i in errored_indices:
            if prediction[i] != selection:
                errors += 1

        cost += params["cluster"]
        return (cost, errors)
    
    cost, errors = 0, 0
    for component in components:
        _cost, _errors = compute(component)
        cost += _cost
        errors += _errors

    return (cost, errors)
