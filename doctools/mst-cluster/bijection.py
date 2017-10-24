
class Bijection:
    def __init__(self):
        self.counter = 0
        self.map = {}
        self.imap = {}

    def image(self, x):
        if x not in self.map:
            self.map[x] = self.counter
            self.imap[self.counter] = x
            self.counter += 1
        return self.map[x]

    def pre_image(self, y):
        if u not in self.imap:
            raise KeyError("Not added to bijection yet")
        return self.imap[y]

    def __call__(self, x):
        return self.image(x)
    


