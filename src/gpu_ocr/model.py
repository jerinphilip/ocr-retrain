import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss

class GravesBatchRNN(nn.Module):
    def __init__(self):
        super(GravesBatchRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=50, hidden_size=50, bidirectional=True, bias=True, batch_first=True)

    def forward(self, x):
        x, _ = self.rnn(x)
        #  _ is hidden states, and cell states?
        x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x

class GravesNN(nn.Module):
    def __init__(self, input_size, output_classes):
        super(GravesNN, self).__init__()
        self.fc_in = nn.Linear(input_size,50)
        self.fc_out = nn.Linear(50, output_classes)
        hidden_ls = [GravesBatchRNN() for i in range(3)]
        self.hidden = nn.Sequential(*hidden_ls)

    def forward(self, x):

        # Reshaping to handle variable length sequences in FC layer.
        batch_size, timesteps = x.size(0), x.size(1)
        x = x.view(batch_size*timesteps, -1)
        x = self.fc_in(x)
        x = x.view(batch_size, timesteps, -1)

        x = self.hidden(x)

        # Reshaping to handle variable length sequences in FC layer.
        batch_size, timesteps = x.size(0), x.size(1)
        x = x.view(batch_size*timesteps, -1)
        x = self.fc_out(x)
        x = x.view(batch_size, timesteps, -1)
        return x


if __name__ == '__main__':
    G = GravesNN(32, 108)
    x = np.random.randn(1, 50, 32)
    x = torch.Tensor(x)
    x = Variable(x, requires_grad=False)
    y = G(x)
    print(y.size())


