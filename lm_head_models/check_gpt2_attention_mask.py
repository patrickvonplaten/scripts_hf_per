#!/usr/bin/env python3

from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers.modeling_gpt2 import GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|endoftext|>')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Complete phrases are: "I like to drink soda without sugar" and "Go watch TV alone, I am not going"
docs = ["I like to drink soda", "Go watch TV"]
docs_tensors = tokenizer.batch_encode_plus(
    [d for d in docs], pad_to_max_length=True, return_tensors='pt')

docs_next = ["without sugar", "alone, I am not going"]
docs_next_tensors = tokenizer.batch_encode_plus(
    [d for d in docs_next], pad_to_max_length=True, return_tensors='pt')

# predicting the first part of each phrase
_, past = model(docs_tensors['input_ids'], attention_mask=docs_tensors['attention_mask'])

# predicting the rest of the phrase
attn_mask = torch.cat([docs_tensors['attention_mask'], docs_next_tensors['attention_mask']], dim=-1)
logits, _ = model(docs_next_tensors['input_ids'], attention_mask=attn_mask, past=past)
logits = logits[:, -1]
_, top_indices_results = logits.topk(30)
