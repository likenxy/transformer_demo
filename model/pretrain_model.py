import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from model.transformer import Transformer
from model.pretrain_head import LMPredictHead, SClsHead

class PretrainModel(nn.Module):
	def __init__(self, config):
		super(PretrainModel, self).__init__()
		self.transformer = Transformer(config)

		self.wm = LMPredictHead(config)
		self.scls = SClsHead(config)

		self.loss_function = CrossEntropyLoss(ignore_index=-1)

		self.config = config

	def forward(self, input_ids, token_type_ids=None, masked_lm_labels = None, next_sentence_labels=None):
		p, e = self.transformer(input_ids, token_type_ids)

		wmp = self.wm(e)
		sclsp = self.scls(p)

		if masked_lm_labels is not None and next_sentence_labels is not None:
			loss1 = self.loss_function(wmp.view(-1, self.config.vocab_size), masked_lm_labels.reshape(-1))
			loss2 = self.loss_function(sclsp.view(-1, 2), next_sentence_labels.view(-1))
			return loss1 + loss2

		return wmp, sclsp