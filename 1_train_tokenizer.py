from tokenizers import (
        models,
        normalizers,
        pre_tokenizers,
        trainers,
        Tokenizer,
        )
import os

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence(
	[normalizers.NFD(), normalizers.Lowercase()]
)
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
#tokenizer.pre_tokenizers = pre_tokenizers.UnicodeScripts()
#tokenizer.pre_tokenizers = pre_tokenizers.BertPreTokenizer()

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=10000, special_tokens=special_tokens)

data_list = []
for root, paths, names in os.walk("./data"):
    for name in names:
        data_list.append(os.path.join(root, name))

tokenizer.train(data_list, trainer=trainer)
tokenizer.save("./model_save/my_tokenizer.json")
