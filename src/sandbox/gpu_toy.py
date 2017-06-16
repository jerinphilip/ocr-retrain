import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.bidirectional:
            x = x.view(x.size(0), 2, -1).sum(2).view(x.size(0), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class GravesLSTM(nn.Module):
    def __init__(self, **kwargs):
        super(GravesLSTM, self).__init__()

        fc_in = nn.Linear(kwargs['input_size'], kwargs['hidden_size'])
        fc_out = nn.Linear(kwargs['hidden_size'], kwargs['output_size'])
        
        rnn = lambda x: nn.LSTM(
                input_size=kwargs['hidden_size'], 
                hidden_size=kwargs['hidden_size'],
                bidirectional=True,
                bias=True)

        rnns = map(rnn, range(kwargs['hidden_count']))

        hiddenLayer = nn.Sequential(*rnns)


        self.layer = {
                "input": fc_in,
                "hidden": hiddenLayer,
                "output": fc_out
        }
    
    def forward(self, x):
        x = self.layer["input"](x)

        x = self.layer["


