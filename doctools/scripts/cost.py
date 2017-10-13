import json
from plot import plot

class Accounting:
    def __init__(self):
        self.U = set()
        self.xs = []
        self.ys = []
        self.review = 0
        self.nreview = 0

    def account(self, **kwargs):
        """
        Receives the computed cost for promoted

        """
        if not self.U:
            self.U = kwargs['excluded']

        self.review += self.cost(kwargs['promoted'])
        remaining = self.cost(kwargs['excluded'])
        projected = self.review + remaining
        self.xs.append(self.nreview)
        self.ys.append(projected)

    def promote(self, **kwargs):
        self.nreview += len(kwargs['indices'])

    def axes(self):
        return (self.xs, self.ys)
    
    def cost(self, ls):
        # ws = [
        #    (False, False, False) 
        #       => Incorrect at all three stages => 30 seconds
        #    (False, False, True)
        #       => Incorrect, but postproc saved => 5 seconds
        #    (False, True)
        #       => Incorrect, but postproc errored => Infinity
        #    (True, False, False)   => Correct => 0 seconds
        #    (True, False, True)    => Correct => 0 seconds
        #    (True, True)           => Correct => 0 seconds
        ws = [30, 5, 0, 0, 0, 0]
        return sum([w*l for w,l in zip(ws, ls)])



def f(path):
    with open(path) as fp:
        stats = json.load(fp)
        methods = []
        for method in stats:
            log = Accounting()
            cexcl = stats[method]["cost"]["excluded"]
            cprom = stats[method]["cost"]["promoted"]
            iprom = stats[method]["index"]["promoted"]
            n = len(cexcl)
            for t in range(n):
                log.promote(indices=iprom[t])
                log.account(excluded=cexcl[t], promoted=cprom[t])
            #print(log.axes())
            methods.append((method, log.axes()))
        plot(methods, path + '.png')

if __name__ == '__main__':
    import sys, os
    output = "/users/jerin/ocr-retrain/output/%s"%(sys.argv[1])
    jsons = filter(lambda f: f.endswith(".json"), os.listdir(output))
    for jsonfile in jsons:
        path = os.path.join(output, jsonfile)
        print(path)
        f(path)
