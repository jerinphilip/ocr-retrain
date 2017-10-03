from copy import deepcopy

class CostModel:
    def __init__(self, error_module=None):
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

    @staticmethod
    def reduce(d, cost):
        r = 0
        for key in cost:
            if type(cost[key]) is dict:
                r += tree_reduce(d[key], cost[key])
            else:
                r += cost[key]*d[key]
        return r

    def account(self, prediction, truth):
        assert(self.em is not None)
        threshold = 1
        ocr = (prediction == truth)
        error = self.em.error(prediction) 
        pp = (error < threshold)
        if not pp: 
            suggest = (truth in self.em.suggest(prediction))
            self.model[ocr][pp][suggest] += 1
        else:
            self.model[ocr][pp] += 1

    def _linear(self, node, path):
        if not isinstance(node, dict):
            return [(path, node)]
        else:
            vs = []
            for key in node:
                if key != "name":
                    paths = self._linear(node[key], tuple(list(path) + [key]))
                    vs.extend(paths)
            return vs

    def linear(self):
        flattened = self._linear(self.model, ())
        flattened.sort()
        headers, values = list(zip(*flattened))
        return (headers, values)

    def export(self):
        return self.model

if __name__ == '__main__':
    cm = CostModel()
    from pprint import pprint
    pprint(cm.linear())

