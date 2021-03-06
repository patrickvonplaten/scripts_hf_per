#!/usr/bin/env python3
import ipdb  # noqa: F401
import random  # noqa: F401
import numpy as np  # noqa: F401
import torch
from argparse import ArgumentParser  # noqa: F401
from transformers import AutoModelWithLMHead, AutoTokenizer


def main(args):
    check_lm_head_loss(args)


def check_lm_head_loss(args):
    tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
    model = AutoModelWithLMHead.from_pretrained('xlnet-base-cased')

    # We show how to setup inputs to predict a next token using a bi-directional context.
    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)).unsqueeze(0)  # We will predict the masked token
    input_ids = torch.cat([input_ids, torch.zeros((1, 1), dtype=torch.long)], dim=1)
    labels = torch.tensor(tokenizer.encode("cute", add_special_tokens=False)).unsqueeze(0)  # We will predict the masked token

    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
    perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token

    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
    target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

    outputs = model(input_ids, labels=labels, perm_mask=perm_mask, target_mapping=target_mapping)
    loss, next_token_logits = outputs[:2]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--input", type=str, default="")
    args = parser.parse_args()
    main(args)
