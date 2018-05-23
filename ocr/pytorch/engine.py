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
from ocr.pytorch.util import gpu_format, load
from .coding import Decoder
from parser.lookup import codebook
import pandas as pd
from ocr.util import cer, wer
import pdb
import math
from ocr.pytorch.model import GravesNet

class Engine:
    def __init__(self, savepath, **kwargs):
        self._kwargs = kwargs
        # self.model = GravesNN(input_size=self._kwargs['input_size'], 
        #         output_classes=self._kwargs['output_classes'])
        self.model = GravesNet(**kwargs)
        print(self.model)
                                
        self.savepath = savepath
        self.optim_params = {
            'lr': 3e-4,
            'momentum': 0.8,
            'nesterov': True
        }
        lookup_filename = 'lookups/Sanskrit.txt'
        self.lmap, self.ilmap = codebook(lookup_filename)
        # pdb.set_trace()
        for key in self.optim_params:
            if key in self._kwargs:
                self.optim_params[key] = self._kwargs[key]
        if 'state_dict' not in self._kwargs:
            self.model.cuda()
            # self.optimizer = optim.SGD(self.model.parameters(), **self.optim_params)
            self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
            self.criterion = CTCLoss()
            self._kwargs['best'] = float("inf")
        else:
            
            self.model.load_state_dict(self._kwargs['state_dict'])
            self._kwargs['best'] = self._kwargs['best']
            self.model.cuda()
            self.optimizer = optim.SGD(self.model.parameters(), **self.optim_params)
            self.criterion = CTCLoss()
            self.start_epoch = self._kwargs['epoch']

    def train_subroutine(self, batch, **kwargs):
        seq, target = batch
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

        if kwargs['validation']:
            er_char, er_word = self.get_accuracy(prediction, target)
            return loss.data[0], er_char, er_word
        return loss.data[0]


    def export(self):
        self._kwargs['state_dict'] = self.model.state_dict()
        return self._kwargs

    
    def get_accuracy(self, prediction, target):
        decoder  = Decoder(self.lmap, self.ilmap)
        predict =  decoder.decode(prediction)
        target = decoder.to_string(target.data)
        words = predict.split(); truths = target.split()
        if len(words)!= len(truths):
            diff = abs(len(truths)-len(words))
            words = words + diff*['']
        er_char = cer(words, truths)
        er_word = wer(words, truths)
        return er_char, er_word

    def save_checkpoint(self, state, is_best):
        """Save checkpoint if a new best is achieved"""
        if is_best:
            print ("=> Saving a new best")
            torch.save(state, self.savepath)  # save checkpoint
            self._kwargs['best'] = state['best']
        else:
            print ("=> Validation Accuracy did not improve")

    def train(self, train_set, **kwargs):
        self.model.train()

        defaults =  {
            'max_epochs': 1000,
            'expected_loss': 35.0,
        }

        for key in defaults:
            if key not in kwargs: kwargs[key] = defaults[key]

        train_split = train_set
        validation_set = train_set
        f = lambda x: x
        if 'debug' in kwargs and kwargs['debug']:
            f = tqdm
        start_epoch = self._kwargs['epoch']
        epoch=start_epoch
        validation_loss = float('inf')
        df = pd.DataFrame(columns=['Train Loss', 'Validation Loss', 'CER', 'WER'])
        while epoch < kwargs['max_epochs']:
            # and validation_loss > kwargs['expected_loss']:

            epoch = epoch + 1
            print('Epochs:[%d]/[%d]'%(epoch, kwargs['max_epochs']))
            train_subset, validation_subset = self._split(train_set)

            # Training
            avgTrain = AverageMeter("train loss")
            for pair in f(train_subset):
                loss = self.train_subroutine(pair, validation=False)
                if not math.isnan(loss) and loss != float('inf'):
                    avgTrain.add(loss)
                else:
                    print(loss)

            print(avgTrain, flush=True)
            # Validation
            avgValidation = AverageMeter("validation loss")
            avgChar = AverageMeter("Character Error Rate")
            avgWord = AverageMeter("Word Error Rate")
            for pair in f(validation_subset):
                loss, er_char, er_word = self.train_subroutine(pair, validation=True)
                if not math.isnan(loss) and loss != float('inf'):
                    avgValidation.add(loss)
                avgChar.add(er_char); avgWord.add(er_word)
            print(avgValidation, flush=True)

            train_loss = avgTrain.compute()
            validation_loss = avgValidation.compute()
            char_error = avgChar.compute()
            word_error = avgWord.compute()
            # Saving state
            print(char_error, word_error)
            df.loc[epoch-start_epoch] = [train_loss, validation_loss, char_error, word_error]
            print(self._kwargs['best'])
            state = validation_loss
            is_best = state < self._kwargs['best']
            self.save_checkpoint({
                            'epoch': epoch,
                            'state_dict': self.model.state_dict(),
                            'best': state
                            }, is_best)
            df.to_csv('loss_log.csv')
        return validation_loss, train_loss
    def _split(self, train_set, **kwargs):
        defaults = {
            'method': 'random',
            'split' : 0.8
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
        print('testing ..')
        avgTest = AverageMeter("test loss")
        for pair in f(test_set):
            loss, er_char, er_word = self.train_subroutine(pair, validation=True)
            avgTest.add(loss)
        print(avgTest, flush=True)

    def recognize(self, sequence, decoder):
        self.model.eval()
        sequence = sequence.cuda()
        sequence = Variable(sequence)
        probs = self.model(sequence)
        prediction = decoder.decode(probs)
        return prediction
