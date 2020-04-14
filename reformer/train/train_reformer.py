#!/usr/bin/env python3
from transformers import ReformerModelWithLMHead, ReformerTokenizer
from transformers import ReformerConfig  # noqa: F401


def get_data_set(path):
    with open(path, 'r') as f:
        text = f.read()

    # The file read above includes metadata and licensing information.
    # For training our language model, we will only use the actual novel text.
    start = text.find('CRIME AND PUNISHMENT')  # skip header
    start = text.find('CRIME AND PUNISHMENT', start + 1)  # skip header
    start = text.find('CRIME AND PUNISHMENT', start + 1)  # skip translator preface
    end = text.rfind('End of Project')  # skip extra text at the end
    text = text[start:end].strip()
    return text


text = get_data_set("reformer_crime-and-punishment-2554.txt")
model = ReformerModelWithLMHead.from_pretrained("patrickvonplaten/reformer-random")
tokenizer = ReformerTokenizer.from_pretrained("patrickvonplaten/reformer-random")

# input_dict['input_ids'].shape = [1, 524288]
input_dict = tokenizer.encode_plus(text, return_tensors='pt', pad_to_max_length=True, max_length=2**19)


import ipdb
ipdb.set_trace()

pass
