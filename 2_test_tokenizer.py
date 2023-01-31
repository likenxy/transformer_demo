from tokenizers import Tokenizer
from util import string_util

tokenizer = Tokenizer.from_file("./model_save/my_tokenizer.json")

s = string_util.inster_space("招财进宝财源广进，紫气东来万寿无疆")

tokens = tokenizer.encode(s)

print(tokens.ids, tokens.tokens)

tokens = tokenizer.encode("")
print(tokens.ids, tokens.tokens)

