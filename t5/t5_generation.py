#!/usr/bin/env python3
from transformers import T5ForConditionalGeneration, TFT5ForConditionalGeneration, T5Tokenizer
import sys

library = sys.argv[1] if len(sys.argv) > 1 else 'pt'

if library == 'pt':
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
elif library == 'tf':
    model = TFT5ForConditionalGeneration.from_pretrained('t5-base')
else:
    raise ValueError('{} does not exist. Use either "tf" or "pt"'.format(library))

tok = T5Tokenizer.from_pretrained('t5-base')
text = "translate English to German: How old are you?"

input_ids = tok.encode(text, return_tensors=library)
outputs = model.generate(input_ids, bos_token_id=tok.pad_token_id, max_length=22, num_beams=4, do_sample=False, early_stopping=True)

print(tok.decode(outputs[0], skip_special_tokens=True))
