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
        self.optimizer = optim.SGD(self.model.parameters(), **self.optim_params)
        self.criterion = CTCLoss()
        for seq, target in inputs:
            # CUDA things
            seq = seq.cuda()
            #target = target.cuda()
            seq = Variable(seq, requires_grad=False)
            target = Variable(target, requires_grad=False)

            # Feedforward
            net_output = self.model(seq)
            #prediction = net_output.transpose(0, 1)
            prediction = net_output

            # Sizes
            #print(prediction.size(1))
            #print(target.size(0))
            target_sizes = Variable(torch.IntTensor([target.size(0)]), 
                    requires_grad=False)
            pred_sizes = Variable(torch.IntTensor([prediction.size(0)]), 
                    requires_grad=False)

            # Compute Loss
            loss = self.criterion(prediction, target, 
                    pred_sizes, target_sizes)
            print(loss.data[0])

            # Backpropogate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(self, inputs):
        raise NotImplementedError

    def validate(self, inputs):
        raise NotImplementedError
