#!/usr/bin/env python3
from transformers import T5ForConditionalGeneration, TFT5ForConditionalGeneration, T5Tokenizer
import sys

library = sys.argv[1]
input_text = sys.argv[2]

size = 't5-base'
if library == 'pt':
    model = T5ForConditionalGeneration.from_pretrained(size)
elif library == 'tf':
    model = TFT5ForConditionalGeneration.from_pretrained(size)
else:
    raise ValueError('{} does not exist. Use either "tf" or "pt"'.format(library))

tok = T5Tokenizer.from_pretrained(size)

input_ids = tok.encode(input_text, return_tensors=library)

#outputs = model.generate(input_ids, bos_token_id=tok.pad_token_id, max_length=50, num_beams=4, do_sample=False, early_stopping=True, length_penalty=2.0)
outputs = model.generate(input_ids, bos_token_id=tok.pad_token_id, max_length=50, num_beams=1, do_sample=False, early_stopping=True, length_penalty=2.0)

print(tok.decode(outputs[0], skip_special_tokens=True))
