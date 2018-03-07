from .dtype import GravesNN
from warpctc_pytorch import CTCLoss
from torch import nn, optim
from torch.autograd import Variable
import torch
from .util import AverageMeter
from random import shuffle
from tqdm import tqdm
from random import shuffle
from math import ceil
import pdb
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
            for key in ['best_model', 'best_state']:
                self._kwargs[key] = kwargs[key]
            self.model.cuda()
            self.optimizer = optim.SGD(self.model.parameters(), **self.optim_params)
            self.criterion = CTCLoss()

    def train_subroutine(self, batch, **kwargs):
        seq, target = batch
        seq = seq.cuda()
        seq = Variable(seq, requires_grad=False)
        target = Variable(target, requires_grad=False)
        # Feedforward
        net_output = self.model(seq)
        prediction = net_output.contiguous()
        target_sizes = Variable(torch.IntTensor([target.size(0)]), 
                requires_grad=False)
        pred_sizes = Variable(torch.IntTensor([prediction.size(0)]), 
                requires_grad=False)

        # Compute Loss
        self.criterion.cuda()
        loss = self.criterion(prediction, target, 
                pred_sizes, target_sizes)

        if not kwargs['validation']:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.data[0]


    def export(self):
        self._kwargs['state_dict'] = self.model.state_dict()
        return self._kwargs
    def train(self, train_set, **kwargs):
        self.model.train()

        defaults =  {
            'max_epochs': 25,
            'expected_loss': 55.0,
        }

        for key in defaults:
            if key not in kwargs: kwargs[key] = defaults[key]

        train_split = train_set
        validation_set = train_set
        f = lambda x: x
        if 'debug' in kwargs and kwargs['debug']:
            f = tqdm
        epoch = 0
        validation_loss = float('inf')
        while epoch < kwargs['max_epochs'] \
                and validation_loss > kwargs['expected_loss']:

            epoch = epoch + 1
            print('Epochs:[%d]/[%d]'%(epoch, kwargs['max_epochs']))
            train_subset, validation_subset = self._split(train_set, method='random')

            # Training
            avgTrain = AverageMeter("train loss")
            for pair in f(train_subset):
                loss = self.train_subroutine(pair, validation=False)
                avgTrain.add(loss)

            print(avgTrain, flush=True)
            # Validation
            avgValidation = AverageMeter("validation loss")
            for pair in f(validation_subset):
                loss = self.train_subroutine(pair, validation=True)
                avgValidation.add(loss)
            print(avgValidation, flush=True)
            train_loss = avgTrain.compute()
            validation_loss = avgValidation.compute()
            # Saving state
            state = (validation_loss, train_loss)
            if epoch+1 % 10 == 0:
                model_ft = self.export()
                torch.save(model_ft, open('file.tar', "wb+"))
            if state < self._kwargs['best']:
                self._kwargs['best_state'] = state
                self._kwargs['best_model'] = self.model.state_dict()
        return validation_loss, train_loss
    def _split(self, train_set, **kwargs):
        defaults = {
            'method': 'sequential',
            'split' : 0.7
        }

        for key in defaults:
            if not key in kwargs: kwargs[key] = defaults[key]

        n = ceil(len(train_set)*kwargs['split'])

        if kwargs['method'] == 'sequential':
            return (train_set[:n], train_set[n:])

        elif kwargs['method'] == 'random':
            shuffle(train_set)
            return (train_set[:n], train_set[n:])

        else:
            print("Unknown method to split train set")
            raise KeyError


    def test(self, test_set, **kwargs):
        f = lambda x: x
        if 'debug' in kwargs and kwargs['debug']:
            f = tqdm

        avgTest = AverageMeter("test loss")
        for pair in f(test_set):
            loss = self.train_subroutine(pair, validation=True)
            avgTest.add(loss)

        print(avgTest, flush=True)


    def recognize(self, sequence):
        self.model.eval()
        sequence = sequence.cuda()
        sequence = Variable(sequence)
        probs = self.model(sequence)
        return probs
