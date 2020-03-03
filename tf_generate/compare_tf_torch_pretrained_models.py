#!/usr/bin/env python3
import ipdb  # noqa: F401
import numpy as np  # noqa: F401
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TFGPT2LMHeadModel
import torch
import tensorflow as tf

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model_pt = GPT2LMHeadModel.from_pretrained('distilgpt2')
model_tf = TFGPT2LMHeadModel.from_pretrained('distilgpt2')

input_string = 'The president'
input_pt = tokenizer.encode(input_string, return_tensors='pt')
input_tf = tokenizer.encode(input_string, return_tensors='tf')
print('input: {}'.format(input_pt))

# tests work
# output_pt = model_pt(input_pt)[0].detach().numpy()
# output_tf = model_tf(input_tf)[0].numpy()
#tf.random.set_seed(0)
#torch.manual_seed(0)

generated_pt = model_pt.generate(input_pt, do_sample=False, bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id)
generated_tf = model_tf.generate(input_tf, do_sample=False, bos_token_id=tokenizer.bos_token_id, eos_token_ids=tokenizer.eos_token_id)

print("PT OUT: {}".format(generated_pt[0]))
print("TF OUT: {}".format(generated_tf[0]))
print("PT: {}".format(tokenizer.decode(generated_pt[0])))
print("TF: {}".format(tokenizer.decode(generated_tf[0])))
assert True
