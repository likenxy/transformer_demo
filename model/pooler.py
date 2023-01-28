import torch
import torch.nn as nn

class Pooler(nn.Module):
	def __init__(self, config):
		super(Pooler, self).__init__()
		self.dense = nn.Linear(config.hidden_dim, config.hidden_dim).to(config.device)
		self.ac = nn.Tanh()

	def forward(self, x):
		f = x[:, 0]
		return self.ac(self.dense(f))