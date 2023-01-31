import torch
import torch.nn as nn

from model.embedding import Embedding
from model.encoder import Encoder
from model.decoder import Decoder
from model.model_config import ModelConfig

class Transformer(nn.Module):
	def __init__(self, config):
		super(Transformer, self).__init__()
		self.embedding = Embedding(config)
		self.encoders = nn.ModuleList(
			[Encoder(config) for _ in range(config.encoder_layer_count)]
		) 
		self.decoders = nn.ModuleList(
			[Decoder(config) for _ in range(config.decoder_layer_count)]
		)
		self.padding_id = config.padding_id
		self.linear = nn.Linear(config.hidden_dim, config.vocab_size).to(config.device)

	def forward(self, src_ids, trg_ids):
		src_embedding = self.embedding(src_ids)
		for encoder in self.encoders:
			src_embedding = encoder(src_embedding)
		
		trg_embedding = self.embedding(trg_ids)
		for decoder in self.decoders:
			trg_embedding = decoder(trg_embedding, src_embedding)
		output = self.linear(trg_embedding)

		return output
