from .model import GravesNN
from parser.webtotrain import read_book
from warpctc_pytorch import CTCLoss
from torch import nn, optim
from torch.autograd import Variable
import torch


def convert(pyocr_output):
    codepoint = pyocr_output[1:]
    codepoint_value = int(codepoint, 16)
    return chr(codepoint_value)

def train(xs, ys, lookup_file):
    labels = None
    criterion = CTCLoss()
    with open(lookup_file, "r") as fp:
        labels = fp.read().splitlines()
        labels = list(map(convert, labels))
        labels = [''] + labels
    vals = list(range(len(labels)))
    labelDict = dict(zip(labels, vals))
    invLabelDict = dict(enumerate(labels))
    model = GravesNN(32, len(labels))
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2,
                momentum=0.2, nesterov=True)
    model.train()
    for x, gt in zip(xs, ys):
        x = x.cuda()
        x = Variable(x, requires_grad=True)
        z = list(map(lambda t: labelDict[t], gt))
        label_sizes = Variable(torch.IntTensor([len(z)]))
        z = Variable(torch.IntTensor(z), requires_grad=False)
        y = model(x)

        print(x.size())
        print(y)

        _, lsize, class_size = y.size()
        y = y.view(lsize, class_size)
        y_sizes = Variable(torch.IntTensor([class_size]))
        loss = criterion(y, z, y_sizes, label_sizes)
        print(loss)
        #loss.backward()
        #optimizer.step()

    return model


