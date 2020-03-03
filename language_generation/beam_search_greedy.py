#!/usr/bin/env python3

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

input_context = 'The dog'
input_ids = torch.tensor(tokenizer.encode(input_context)).unsqueeze(0)
outputs = model.generate(input_ids=input_ids, num_beams=10, num_return_sequences=3, do_sample=False)
for i in range(3):
    print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))
