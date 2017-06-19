from .dtype import GravesNN
from warpctc_pytorch import CTCLoss
from torch import nn, optim
from torch.autograd import Variable
import torch
from .util import AverageMeter

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

    def train_subroutine(self, seq_targ, **kwargs):
        seq, target = seq_targ
        seq = seq.cuda()
        #target = target.cuda()
        seq = Variable(seq, requires_grad=False)
        target = Variable(target, requires_grad=False)

        # Feedforward
        net_output = self.model(seq)
        #prediction = net_output.transpose(0, 1)
        prediction = net_output

        # Sizes
        target_sizes = Variable(torch.IntTensor([target.size(0)]), 
                requires_grad=False)
        pred_sizes = Variable(torch.IntTensor([prediction.size(0)]), 
                requires_grad=False)

        # Compute Loss
        loss = self.criterion(prediction, target, 
                pred_sizes, target_sizes)

        # Backpropogate
        if not kwargs['validation']:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.data[0]

    def train(self, train_set, validation_set):
        self.model.cuda()
        self.optimizer = optim.SGD(self.model.parameters(), **self.optim_params)
        self.criterion = CTCLoss()
        satisfactory = False

        while not satisfactory:
            avgTrain = AverageMeter("train loss")
            for pair in train_set:
                loss = self.train_subroutine(pair, validation=False)
                avgTrain.add(loss)
            print(avgTrain)
            avgValidation = AverageMeter("validation loss")
            for pair in validation_set:
                loss = self.train_subroutine(pair, validation=True)
                avgValidation.add(loss)
            print(avgValidation)

    def test(self, inputs):
        raise NotImplementedError

    def validate(self, inputs):
        raise NotImplementedError
