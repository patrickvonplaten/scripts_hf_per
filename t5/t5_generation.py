#!/usr/bin/env python3
from transformers import T5WithLMHeadModel, T5Tokenizer

model = T5WithLMHeadModel.from_pretrained('t5-base')
tok = T5Tokenizer.from_pretrained('t5-base')

text = "translate English to German: How old are you?"

input_ids = tok.encode(text, return_tensors='pt')
outputs = model.generate(input_ids, bos_token_id=tok.pad_token_id, max_length=22, num_beams=4, do_sample=False, early_stopping=True)

print(tok.decode(outputs[0], skip_special_tokens=True))
