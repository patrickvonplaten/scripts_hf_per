#!/usr/bin/env python3 

from transformers import BertTokenizer
batch_input_str = (("Mary spends $20 on pizza"), ("She likes eating it"), ("The pizza was great"))
input_str = ['hello', 'hello', 'hey']
tok = BertTokenizer.from_pretrained('bert-base-uncased')

import ipdb
ipdb.set_trace()
print(tok.batch_encode_plus(batch_input_str, pad_to_max_length=True))
