#!/usr/bin/env python3

from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

model = TFGPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
inputs_dict = tokenizer.encode_plus('The dog', return_tensors='tf')
outputs = model.generate(inputs_dict['input_ids'], num_beams=10, max_length=20, num_return_sequences=3)
for output in outputs:
    print(tokenizer.decode(output, skip_special_tokens=True))
