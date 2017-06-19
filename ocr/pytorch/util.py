import torch
import numpy as np

def gpu_format(label_map):
    def ocr_ready(seq_targ):
        seq, targ = seq_targ
        seq = torch.Tensor(np.array([seq], dtype=np.float32))
        # The above generates BxHxT - Convert to TxBxH
        seq = seq.permute(2, 0, 1).contiguous() 
        targ = list(map(lambda x: label_map[x], targ))
        targ = torch.IntTensor(targ)
        return (seq, targ)
    return ocr_ready


class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.total = 0

    def add(self, element):
        self.total += element
        self.count += 1

    def compute(self):
        if self.count == 0:
            return float("inf")
        return self.total/self.count
    
    def __str__(self):
        return "Average %s: %.6lf"%(self.name, self.compute())
