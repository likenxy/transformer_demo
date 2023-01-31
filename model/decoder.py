import torch
import torch.nn as nn

from model.multi_head_attention import MultiHeadAttention
from model.model_config import ModelConfig

class Decoder(nn.Module):
	def __init__(self, config):
		super(Decoder, self).__init__()
		self.attention = MultiHeadAttention(config)
		self.encode_decode_attention = MultiHeadAttention(config)

	def forward(self, x, encoder_out):
		x = self.attention(x)
		x = self.encode_decode_attention(
			x, encoder_out
		)
		return x
