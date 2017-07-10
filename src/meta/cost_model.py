from copy import deepcopy

class CostModel:
    def __init__(self, error_module):
        self.em = error_module
        suggest = {
            True: 0,
            False: 0,
            "name": "suggest"
        }

        postproc = {
                True: 0,
                False: deepcopy(suggest),
                "name": "postproc"
        }

        ocr = {
            True: deepcopy(postproc),
            False: deepcopy(postproc),
            "name": "ocr"
        }

        self.model = ocr

    def account(self, prediction, truth):
        threshold = 1
        ocr = (prediction == truth)
        error = self.em.error(prediction) 
        pp = (error < threshold)
        if not pp: 
            suggest = (truth in self.em.suggest(prediction))
            self.model[ocr][pp][suggest] += 1
        else:
            self.model[ocr][pp] += 1

    def export(self):
        return self.model
