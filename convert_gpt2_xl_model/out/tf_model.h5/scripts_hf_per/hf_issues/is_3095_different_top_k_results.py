#!/usr/bin/env python3

from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers.modeling_gpt2 import GPT2LMHeadModel
import torch
import ipdb

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|endoftext|>')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Complete phrases are: "I like to drink soda and" and "Please help me with this"
docs = ["I like to", "Please help me"]
# note: comment the above line and uncomment the following line to make it work with 1 document
#docs = ["I like to"]
docs_tensors = tokenizer.batch_encode_plus(
    [d for d in docs], pad_to_max_length=True, return_tensors='pt')

docs_next = ["soda and ", "with this"]
# note: comment the above line and uncomment the following line to make it work with 1 document
#docs_next = ["soda and "]
docs_next_tensors = tokenizer.batch_encode_plus(
    [d for d in docs_next], pad_to_max_length=True, return_tensors='pt')

# predicting the first part of each phrase
_, past = model(docs_tensors['input_ids'], attention_mask=docs_tensors['attention_mask'])

# predicting the rest of the phrase
attn_mask = torch.cat([docs_tensors['attention_mask'], docs_next_tensors['attention_mask']], dim=-1)
logits, _ = model(docs_next_tensors['input_ids'], attention_mask=attn_mask, past=past)
logits = logits[:, -1]
_, top_indices_results = logits.topk(50)

words = [tokenizer.decode([idx.item()]) for tir in top_indices_results for idx in tir]

print("Results with past:", words)

#####################
docs_full_tensors = tokenizer.batch_encode_plus(
    [d + n for d, n in zip(docs, docs_next)], pad_to_max_length=True, return_tensors='pt')
logits, _ = model(docs_full_tensors['input_ids'], attention_mask=docs_full_tensors['attention_mask'])

print('Concat attn_mask: {}'.format(attn_mask))
print('Full attn_mask: {}'.format(docs_full_tensors['attention_mask']))
ipdb.set_trace()

logits = logits[:, -1]
_, top_indices_results = logits.topk(50)

words = [tokenizer.decode([idx.item()]) for tir in top_indices_results for idx in tir]

print("Results without past:", words)
