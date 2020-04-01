#!/usr/bin/env python3

from transformers import GPT2Model, GPT2Tokenizer
import torch

model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<PAD>')

input_ids = tokenizer.encode("Hello, how are you?", return_tensors='pt')
inputs_embeds = model.wte(input_ids)

model(inputs_embeds=inputs_embeds)  # runs without error
