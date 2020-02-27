#!/usr/bin/env python3

from transformers import GPT2LMHeadModel
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')

input_ids = torch.tensor([8, 8, 0, 50256, 50256]).unsqueeze(0)
attn_mask = torch.tensor([1, 1, 1, 0, 0]).unsqueeze(0)

# first step there is no past so lets get it from the model and append new embedding id to inputs and extend the attn_mask
logits_output, past = model(input_ids, attention_mask=attn_mask)
next_token = torch.argmax(logits_output[:, -1, :]).unsqueeze(0)
input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1) 
attn_mask = torch.cat([attn_mask, torch.ones((attn_mask.shape[0], 1)).long()], dim=1)


# now we have a past so we can use it to speed up training
model_inputs = model.prepare_inputs_for_generation(input_ids=input_ids, past=past)
input_ids, past = model(**model_inputs, attention_mask=attn_mask)  # this leads to an error which it should not
