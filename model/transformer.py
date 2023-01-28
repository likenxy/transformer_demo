import torch
import torch.nn as nn

from model.multi_head_attention import MultiHeadAttention
from model.embedding import Embedding
from model.model_config import ModelConfig
from model.pooler import Pooler

class Transformer(nn.Module):
	def __init__(self, config):
		super(Transformer, self).__init__()
		self.embedding = Embedding(config)
		self.layers = nn.ModuleList([
			MultiHeadAttention(config) for _ in range(config.layer_count)
		])
		self.pooler = Pooler(config)
		self.device = config.device

	def forward(self, input_ids, token_type_ids=None):
		em = self.embedding(input_ids, token_type_ids)
		for layer in self.layers:
			em = layer(em)
		p = self.pooler(em)
		return p, em