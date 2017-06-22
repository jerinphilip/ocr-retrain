from .dtype import GravesNN
from warpctc_pytorch import CTCLoss
from torch import nn, optim
from torch.autograd import Variable
import torch
from .util import AverageMeter
from random import shuffle
from tqdm import tqdm

class Engine:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
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

        if 'state_dict' not in kwargs:
            self.model.cuda()
            self.optimizer = optim.SGD(self.model.parameters(), **self.optim_params)
            self.criterion = CTCLoss()
            self._kwargs['best'] = (float("inf"), float("inf"))
        else:
            self.model.load_state_dict(kwargs['state_dict'])
            self.model.cuda()
            self.optimizer = optim.SGD(self.model.parameters(), **self.optim_params)
            self.criterion = CTCLoss()

    def train_subroutine(self, batch, **kwargs):
        #seq, target, psizes, lsizes = batch
        seq, target = batch
        seq = seq.cuda()
        seq = Variable(seq, requires_grad=False)
        target = Variable(target, requires_grad=False)

        # Feedforward
        net_output = self.model(seq)
        prediction = net_output.contiguous()
    
        # Sizes
        target_sizes = Variable(torch.IntTensor([target.size(0)]), 
                requires_grad=False)
        pred_sizes = Variable(torch.IntTensor([prediction.size(0)]), 
                requires_grad=False)

        # Compute Loss
        self.criterion.cuda()
        loss = self.criterion(prediction, target, 
                pred_sizes, target_sizes)
        # print("Loss:", loss.data[0])
        # Backpropogate
        if not kwargs['validation']:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.data[0]


    def export(self):
        self._kwargs['state_dict'] = self.model.state_dict()
        return self._kwargs

    def train(self, train_set, validation_set, **kwargs):
        self.model.train()
        f = lambda x: x
        if 'debug' in kwargs and kwargs['debug']:
            f = tqdm

        avgTrain = AverageMeter("train loss")
        for pair in f(train_set):
            loss = self.train_subroutine(pair, validation=False)
            avgTrain.add(loss)
        avgValidation = AverageMeter("validation loss")
        for pair in f(validation_set):
            loss = self.train_subroutine(pair, validation=True)
            avgValidation.add(loss)

        print(avgTrain)
        state = (avgValidation.compute(), avgTrain.compute())
        if state < self._kwargs['best']:
            self._kwargs['best'] = state
            self._kwargs['best_state'] = self.model.state_dict()

        print(avgValidation)
        return state


    def test(self, test_set):
        raise NotImplementedError

    def recognize(self, sequence):
        self.model.eval()
        sequence = sequence.cuda()
        sequence = Variable(sequence)
        probs = self.model(sequence)
        return probs
