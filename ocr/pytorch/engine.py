from .dtype import GravesNN
from warpctc_pytorch import CTCLoss
from torch import nn, optim
from torch.autograd import Variable
import torch


class Engine:
    def __init__(self, **kwargs):
        self.model = GravesNN(input_size=kwargs['input_size'], 
                output_classes=kwargs['output_classes'])


        self.optim_params = {
            'lr': 3e-4,
            'momentum': 0.8,
            'nesterov': True
        }
        for key in self.optim_params:
            if key in kwargs:
                self.optim_params[key] = kwargs[key]


    def train(self, inputs):
        self.model.cuda()
        self.optim = optim.SGD(self.model.parameters(), **self.optim_params)
        self.criterion = CTCLoss()
        for seq, target in inputs:
            # CUDA things
            seq = seq.cuda()
            #target = target.cuda()
            seq = Variable(seq)
            target = Variable(target, requires_grad=False)

            # Feedforward
            net_output = self.model(seq)
            prediction = net_output.transpose(0, 1)

            # Sizes
            target_sizes = Variable(torch.IntTensor([target.size(0)]))
            pred_sizes = Variable(torch.IntTensor([prediction.size(1)]))

            # Compute Loss
            loss = self.criterion(prediction, target, pred_sizes, target_sizes)
            print(loss.data[0])

            # Backpropogate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
