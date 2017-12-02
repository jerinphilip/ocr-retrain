from .params import params


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
