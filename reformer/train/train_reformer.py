#!/usr/bin/env python3
from transformers import ReformerModelWithLMHead, ReformerTokenizer
import nlp


model = ReformerModelWithLMHead.from_pretrained("patrickvonplaten/reformer-crime-and-punish")
tokenizer = ReformerTokenizer.from_pretrained("patrickvonplaten/reformer-crime-and-punish")

model.to("cuda")
data = nlp.load("crime_and_punish")

input_text = '\n'.join(data['train']['paragraph'][:100])

input_ids = tokenizer.encode(input_text, return_tensors="pt")
input_ids = input_ids.to("cuda")

loss = model(input_ids, lm_labels=input_ids)[0]


pass
