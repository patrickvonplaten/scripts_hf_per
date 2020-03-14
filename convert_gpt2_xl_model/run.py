#!/usr/bin/env python3
from transformers import GPT2LMHeadModel, TFGPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
model.save_pretrained('./')
model = TFGPT2LMHeadModel.from_pretrained('./', from_pt=True)
model.save_pretrained('./out')
