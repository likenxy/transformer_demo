from tokenizers import (
        models,
        normalizers,
        pre_tokenizers,
        trainers,
        Tokenizer,
        )

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence(
	[normalizers.NFD(), normalizers.Lowercase()]
)
tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=10000, special_tokens=special_tokens)

tokenizer.train(["./data/1.txt", "./data/2.txt"], trainer=trainer)
tokenizer.save("./model_save/my_tokenizer.json")