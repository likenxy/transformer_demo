from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("./model_save/my_tokenizer.json")
tokens = tokenizer.encode("我喜欢看刘慈欣的三体小说")

print(tokens.ids, tokens.tokens)