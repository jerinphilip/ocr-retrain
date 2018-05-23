from torch import nn


class TimeDistributedDense(nn.Module):
    def __init__(self, module):                                                 
        super(TimeDistributedDense, self).__init__()                          
        self.module = module

    def forward(self, x):
        timesteps, batch_size = x.size(0), x.size(1)                    
        x = x.view(batch_size*timesteps, -1)                            
        x = self.module(x)                                               
        x = x.view(timesteps, batch_size, -1)                           
        return x

class GravesBatchRNN(nn.Module):                                        
    def __init__(self, **kwargs):                                                 
        super(GravesBatchRNN, self).__init__()                          
        default = {
                'input_size': 50,
                'hidden_size': 50,
                'bidirectional': True,
                'bias': True
        }
        for key in default:
            if key not in kwargs:
                kwargs[key] = default[key]

        self.rnn = nn.LSTM(**kwargs)
                                                                        
    def forward(self, x):                                               
        x, _ = self.rnn(x)                                              
        #  _ is hidden states, and cell states?                         
        x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x                                                        

class GravesNN(nn.Module):
    def __init__(self, **kwargs):
        super(GravesNN, self).__init__()

        default = { 'hidden_size': 50, 'depth': 3 }
        for key in default:
            if key not in kwargs:
                kwargs[key] = default[key]

        fc_in = nn.Linear(kwargs['input_size'], kwargs['hidden_size'])
        fc_out = nn.Linear(kwargs['hidden_size'], kwargs['output_classes'])

        fc_in = TimeDistributedDense(fc_in)
        fc_out = TimeDistributedDense(fc_out)

        hidden_layers = [GravesBatchRNN() for i in range(kwargs['depth'])]             
        self.module = nn.Sequential(fc_in, *hidden_layers, fc_out)

    def forward(self, x):                                               
        return self.module(x)