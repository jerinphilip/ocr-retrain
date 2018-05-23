from torch import nn
import torch
import pdb

class SimpleLinear(nn.Module):
	def __init__(self, module):                                                 
		super(SimpleLinear, self).__init__()                          
		self.module = module

	def forward(self, x):
		
		timesteps, batch_size = x.size(0), x.size(1)                    
		x = x.view(batch_size*timesteps, -1)                           
		x = self.module(x)                                               
		x = x.view(timesteps, batch_size, -1)                           
		return x

class SimpleLSTM(nn.Module):
	def __init__(self, **kwargs):
		super(SimpleLSTM, self).__init__()


		# for key in default:
		# 	if key not in kwargs:
		# 		kwargs[key] = default[key]
	
		self.rnn = nn.LSTM(
						kwargs['hidden_size'], 
						kwargs['hidden_size'], 
						bidirectional = True,
						batch_first = True
						)
		# self.linearLayer = nn.Linear(
		# 				kwargs['hidden_size']*2, 
		# 				kwargs['output_classes']
		# 					)
		# self.softmax = softmax()

	def forward(self, x):
		# x has dimension T * B * D

		T, B, D = x.size(0), x.size(1), x.size(2)
		x = torch.transpose(x, 1, 0)
		x, _ = self.rnn(x) #x has three dimensions batchSize* seqLen * HiddenDim
		x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
		x = torch.transpose(x, 1, 0)
		x = x.contiguous()
		return x
 

class GravesNet(nn.Module):
	def __init__(self, **kwargs):
		super(GravesNet, self).__init__()

		default = { 'hidden_size': 50, 'depth': 3 }
		for key in default:
			if key not in kwargs:
				kwargs[key] = default[key]

		self.fc_in = nn.Linear(kwargs['input_size'], kwargs['hidden_size'])
		self.fc_out = nn.Linear(kwargs['hidden_size'], kwargs['output_classes'])
		self.fc_in = SimpleLinear(self.fc_in)
		self.fc_out = SimpleLinear(self.fc_out)

		self.hidden_layers = [SimpleLSTM(**kwargs)for i in range(kwargs['depth'])]
		self.module = nn.Sequential(self.fc_in, *self.hidden_layers, self.fc_out)
	def forward(self, x):
		# x has dimension T * B * D
		x = self.module(x)
		T, B, D = x.size(0), x.size(1), x.size(2)
		x = x.view(T*B, -1)
		out = nn.functional.softmax(x)
		return out.view(T, B, -1)