#!/usr/bin/env python3

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

inputs_dict = tokenizer.encode_plus('The dog', return_tensors='pt')
outputs = model.generate(inputs_dict['input_ids'], num_beams=3, max_length=10)

# printed sorted_hyps from line: 1088
# Beam_idx: 1 - tensor([ 464, 3290,  635,  468,  257, 2041, 3895,  326,  481, 1037])
# Beam_idx: 2 - tensor([ 464, 3290,  635,  468,  257, 2041, 3895,  326,  481, 1037])
# Beam_idx: 3 - tensor([ 464, 3290,  635,  468,  257, 2041, 3895,  326,  481, 1037])

print("Generated: {}".format(tokenizer.decode(outputs[0])))  # Generated: The dog, named T.H., was recently


# printed sorted_hyps from line: 1088
# Beam_idx: 1 - tensor([ 464, 3290,  373,  788, 3888,  284,  257, 6716, 2156,  351])
# Beam_idx: 2 - tensor([ 464, 3290,  373,  788, 3888,  284,  257, 6716, 2156,  11])
# Beam_idx: 3 - tensor([ 464, 3290,  373,  788, 3888,  284,  257, 6716, 2156,  1566])

print("Generated: {}".format(tokenizer.decode(outputs[0])))  # Generated: The dog was then moved to a nearby house until
