#!/usr/bin/env python3

from transformers import EncoderDecoderModel, BertTokenizer


model = EncoderDecoderModel.from_pretrained('bert-base-uncased', 'bert-base-uncased')

tok = BertTokenizer.from_pretrained('bert-base-uncased')
input_ids = tok.encode('Hi it is me.', return_tensors='pt')

output = model.generate(input_ids, bos_token_id=tok.pad_token_id)

print(tok.decode(output[0], skip_special_tokens=True))
import ipdb
ipdb.set_trace()
pass
