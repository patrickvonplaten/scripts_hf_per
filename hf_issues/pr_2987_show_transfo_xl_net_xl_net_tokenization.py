#!/usr/bin/env python3

from transformers import AutoTokenizer


def show_tokenization(text, model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    print(tok.decode(tok.encode(text, add_special_tokens=False)))


show_tokenization('This is an example. See what happens with, and. ?', 'transfo-xl-wt103')
show_tokenization('This is an example. See what happens with, and. ?', 'xlnet-base-cased')

# prints:
# You might want to consider setting `add_space_before_punct_symbol=True` as an argument to the `tokenizer.encode()` to avoid tokenizing words with punctuation symbols to the `<unk>` token
# This is an <unk> See what happens <unk> <unk>?
# This is an example. See what happens with, and.?
