import torch
import json

class ModelConfig:
	def __init__(self,
				 device = torch.device("cpu"),
				 vocab_size = None,
				 hidden_dim = None,
				 n_head = None,
				 drop_out = None,
				 encoder_layer_count = None,
				 decoder_layer_count = None,
				 max_len = None,
				 padding_id = None):
		self.device = device
		self.vocab_size = vocab_size
		self.hidden_dim = hidden_dim
		self.n_head = n_head
		self.drop_out = drop_out
		self.encoder_layer_count = encoder_layer_count
		self.decoder_layer_count = decoder_layer_count
		self.max_len = max_len
		self.padding_id = padding_id
		self.use_gpu = (self.device == torch.device("cuda"))

	def save(self, path):
		f = open(path, "w")
		d = {
			"vocab_size":self.vocab_size,
			"hidden_dim":self.hidden_dim,
			"n_head":self.n_head,
			"drop_out":self.drop_out,
			"encoder_layer_count":self.encoder_layer_count,
			"decoder_layer_count":self.decoder_layer_count,
			"max_len":self.max_len,
			"padding_id":self.padding_id,
		}
		d = json.dumps(d)
		f.write(d)
		f.close()

	def load(self, path):
		d = open(path).read()
		d = json.loads(d)
		self.vocab_size = d['vocab_size']
		self.hidden_dim = d['hidden_dim']
		self.n_head = d['n_head']
		self.drop_out = d['drop_out']
		self.encoder_layer_count = d['encoder_layer_count']
		self.decoder_layer_count = d['decoder_layer_count']
		self.max_len = d['max_len']
		self.padding_id = d['padding_id']

