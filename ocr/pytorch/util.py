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
