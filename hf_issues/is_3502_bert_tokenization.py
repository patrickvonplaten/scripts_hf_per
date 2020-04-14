#!/usr/bin/env python3
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

problematic_string = "16."

tokens = bert_tokenizer.tokenize(problematic_string)

print(bert_tokenizer.batch_encode_plus([tokens], is_pretokenized=True))
print(bert_tokenizer.batch_encode_plus([problematic_string], is_pretokenized=True))
print(bert_tokenizer.encode_plus(problematic_string))
print(bert_tokenizer.encode_plus(tokens))
