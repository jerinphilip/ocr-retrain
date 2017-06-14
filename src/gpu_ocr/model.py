from .dsRip import BatchRNN
import torch.nn as nn

def create_network(**kwargs):
    hidden_size = kwargs['hidden_size']
    input_size = kwargs['input_size']
    output_classes = kwargs['output_classes']
    hidden_depth = kwargs['hidden_depth']

    def create_hidden(i):
        parameters = {
                "input_size": hidden_size,
                "hidden_size": hidden_size,
                "rnn_type": nn.LSTM,
                "bidirectional": True,
                "batch_norm": False
        }
        rnn = BatchRNN(**parameters)
        return rnn

    rnns = list(map(create_hidden, range(hidden_depth)))

    inputLayer = nn.Linear(input_size, hidden_size)
    outputLayer = nn.Linear(hidden_size, output_classes)

    layers = [inputLayer] + rnns + [outputLayer]

    net = nn.Sequential(*layers)
    return net

