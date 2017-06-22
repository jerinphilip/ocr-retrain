import torch

class Decoder:

    def __init__(self, lmap, ilmap):
        self.lmap = lmap
        self.ilmap = ilmap

    def to_string(self, indices):
        chars = list(map(lambda x: self.ilmap[x], indices))
        string = ''.join(chars)
        return string


    def decode(self, probs):
        """ Convert a probability matrix to sequences """
        _, max_probs = torch.max(probs.transpose(0, 1), 2)
        max_probs = max_probs.squeeze()
        #print(max_probs.size())
        ls = max_probs.data.tolist()
        #print(self.to_string(ls))
        pass
