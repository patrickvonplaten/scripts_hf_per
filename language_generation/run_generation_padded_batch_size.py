#!/usr/bin/env python3

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<PAD>')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='right', pad_token='<PAD>')
# IMPORTANT: Note that setting the <PAD> token like this itn the constructor gives the
# pad_token the pad_token_id = 50256, which normally belongs to <BOS> token_ids in GPT2
# This is a very ugly way that works at the moment of setting the pad_token_id to the <BOS> token that is already included in the vocab size. This will be updated in the coming weeks! # noqa: E501

prompt_text = [
    'in this paper we',
    'we are trying to',
    'The purpose of this workshop is to check whether we can']

# encode plus batch handles multiple batches and automatically creates attention_masks
seq_len = 11
encodings_dict = tokenizer.batch_encode_plus(prompt_text, max_length=seq_len, pad_to_max_length=True)

# ideally we should be able to just input the following two variables to the function model.generate() ... => to be implemented soon!  # noqa: E501
input_ids = torch.tensor(encodings_dict['input_ids'])
attn_mask = torch.tensor(encodings_dict['attention_mask'])

num_tokens_to_produce = 20
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id
eos_not_in_sents = torch.ones(input_ids.shape[0]).long()

# we need to get the token ids of the last non-padded value
last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
start_idx = inp_idx = (last_non_masked_idx).view(-1, 1).repeat(1, tokenizer.vocab_size).unsqueeze(1)
past = None

# get correct position ids
position_ids = torch.tensor([list(range(seq_len)) for i in range(input_ids.shape[0])])
for i, position_ids_slice in enumerate(position_ids):
    position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]]

for step in range(num_tokens_to_produce):
    outputs = model(input_ids, attention_mask=attn_mask, position_ids=position_ids)

    # in the first decoding step, we want to use the 'real' last position for each sentence
    if step == 0:
        next_token_logits = outputs[0].gather(1, start_idx).squeeze(1)
    else:
        next_token_logits = outputs[0][:, -1, :]

    next_tokens = torch.argmax(next_token_logits, dim=-1)

    # this updates which sentences have not seen an <EOS> token so far
    # if one <EOS> token was seen the sentence is finished
    eos_not_in_sents.mul_(next_tokens.ne(eos_token_id).long())

    # either append a padding token here if <EOS> has been seen or append next token
    tokens_to_add = next_tokens * (eos_not_in_sents) + pad_token_id * (1 - eos_not_in_sents)

    # Update input_ids, attn_mask and position_ids
    input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
    attn_mask = torch.cat([attn_mask, torch.ones((attn_mask.shape[0], 1)).long()], dim=1)
    position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

[print(tokenizer.decode(output, skip_special_tokens=True)) for output in input_ids]
