import random

from model.model_config import ModelConfig
from model.transformer import Transformer
from tokenizers import Tokenizer
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os

class CoupleDataset(Dataset):
    def __init__(self, path):
        self.tokenizer = Tokenizer.from_file("./model_save/my_tokenizer.json")
        in_data = open(os.path.join(path, "in.txt")).readlines()
        out_data = open(os.path.join(path, "out.txt")).readlines()
        assert len(in_data) == len(out_data)
        # 生成数据
        data = [] 
        for i in tqdm(range(len(in_data))):
            s, t = in_data[i][:30], out_data[i][:30]
            if len(s) == 0 or len(t) == 0:
                continue
            sids = self.tokenizer.encode(s).ids
            tids = self.tokenizer.encode(t).ids
            size = len(tids)
            for j in range(size):
                data.append([sids, [3] + tids[:size - j]])
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        s, t = self.data[index]
        if len(s) < 30:
            s = s + [1 for _ in range(30 - len(s))]
        if len(t) < 31:
            t = t + [1 for _ in range(31 - len(t))]
        return np.array(s), np.array(t) 
        

def train():
	config = ModelConfig()
	config.load("./model_save/config.json")
	model = Transformer(config)

	data = CoupleDataset("./data/test")
	loader = DataLoader(data, batch_size=20, shuffle=True)
	optimizer = optim.Adam(model.parameters(), lr=1e-4)

	criterion = torch.nn.CrossEntropyLoss(ignore_index=config.padding_id)
	epoch = 20
	step = 0
	for e in range(epoch):
		pbar = tqdm(loader)
		for s, t in pbar:
			optimizer.zero_grad()
			p = model(s, t[:,:-1])
			pv = p.view(-1, p.shape[-1])
			y = t[:,1:].contiguous().view(-1)
			loss = criterion(pv, y)
			loss.backward()
			optimizer.step()
			desc = '[{}/{}][{}][{}]'.format(e+1, epoch, step, float(loss))
			pbar.set_description(desc)
			step += 1
			if step % 300 == 0:
				print(p.size())
				pids = torch.argmax(p, dim=2)
				y = t[:,1:]
				for i in range(p.size(0)):
					ss = data.tokenizer.decode(s.detach().cpu().numpy()[i])
					p_s = data.tokenizer.decode(pids.detach().cpu().numpy()[i])
					ori_s = data.tokenizer.decode(y.detach().cpu().numpy()[i])
					print("\n\n==[{}]==\n==[{}]==\n==[{}]==\n\n".format(ss, p_s, ori_s))
			if step % 2000 == 0:
				torch.save(model.state_dict(), "./model_save/c_model_{}.pt".format(step))
		torch.save(model.state_dict(), "./model_save/c_model_done.pt".format(step))

if __name__ == "__main__":
	train()
 
