import torch
import numpy as np
from collections import namedtuple

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


def bucket(data, **kwargs):
    max_size = 32768
    if 'max_size' in kwargs:
        max_size = kwargs['max_size']
    seqs, truths = zip(*data)
    widths = list(map(lambda x: x.size(0), seqs))
    lsizes = list(map(lambda y: y.size(0), truths))
    # print(widths)
    print("Width (mu, sigma) = (%lf, %lf)"%(np.mean(widths), np.std(widths)))
    print("Lsize (mu, sigma) = (%lf, %lf)"%(np.mean(lsizes), np.std(lsizes)))

    # Naive Greedy Algorithm?

    input_size = seqs[0].size(2)
    start, end = 0, 0 
    batch_size, width = 1, widths[end] 
    swidth = lsizes[end]
    params = []
    while end < len(data):
        if end + 1 == len(data):
            params.append((start, end, batch_size, width, swidth))
            break

        prospective_width = max(width, widths[end+1])
        prospective_size = prospective_width * (batch_size + 1) * input_size
        if prospective_size > max_size:
            params.append((start, end, batch_size, width,swidth))
            start = end+1
            end = end+1
            batch_size, width = 1, widths[end]
            swidth = lsizes[end]
        else:
            end = end+1
            width = prospective_width
            swidth = max(swidth, lsizes[end])
            batch_size += 1


    batches = []

    def pad(tensor, length):
        """ Pads a 2D tensor with zeros."""
        W, H = tensor.size()
        return torch.cat([tensor, tensor.new(length-W, H).zero_()])

    def pad1d(tensor, length):
        W = tensor.size(0)
        return torch.cat([tensor, tensor.new(length-W).zero_()])

    def to_2d(tensor_3d):
        """ TXBXH -> TxH """
        T, _, H = tensor_3d.size()
        return tensor_3d.squeeze(1)

    for s, e, bs, w, sw in params:
        batch = []
        sbatch = []
        for i in range(s, e+1):
            tmp = pad(to_2d(seqs[i]), w)
            tmp = tmp.unsqueeze(1) # get this entry to 3D
            batch.append(tmp)

            tmpseq = pad1d(truths[i], sw)
            tmpseq = tmpseq.unsqueeze(0) # to 2D
            sbatch.append(tmpseq)

        batch_element = torch.cat(batch, 1)
        sbatch_element = torch.cat(sbatch, 0)
        ws = torch.IntTensor(widths[s:e+1])
        ls = torch.IntTensor(lsizes[s:e+1])
        batches.append((batch_element, sbatch_element, 
            ws, ls))
        print(batch_element.size(), sbatch_element.size(), ws, ls)
    return batches




class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.total = 0
        self.max = -1*float("inf")
        self.min = float("inf")

    def add(self, element):
        self.total += element
        self.count += 1
        self.max = max(self.max, element)
        self.min = min(self.min, element)

    def compute(self):
        if self.count == 0:
            return float("inf")
        return self.total/self.count
    
    def __str__(self):
        return "%s (min, max, avg): (%.3lf, %.3lf, %.3lf)"%(self.name, self.max, self.min, self.compute())
