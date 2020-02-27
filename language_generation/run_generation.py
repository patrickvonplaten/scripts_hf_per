#!/usr/bin/env python3
import ipdb  # noqa: F401
import random  # noqa: F401
import numpy as np  # noqa: F401
import torch
from argparse import ArgumentParser  # noqa: F401
from transformers import AutoModelWithLMHead, AutoTokenizer


def main(args):
    run_generation(args)


TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. """


def run_generation(args):
    model = AutoModelWithLMHead.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.input:
        input_text = args.input
    else:
        input_text = TEXT

    print("Text: {}".format(input_text))
    tokenized_input_words = (
        torch.tensor(tokenizer.encode(input_text, add_special_tokens=False)).unsqueeze(0)
    )
    print("BOS: {}".format(tokenizer.bos_token_id))
    print("PAD: {}".format(tokenizer.pad_token_id))
    print("EOS: {}".format(tokenizer.eos_token_id))
    print("Input tokens: {}".format(tokenized_input_words))
    torch.manual_seed(0)
    generated_tokens = model.generate(
        tokenized_input_words,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_ids=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    print("Output tokens: {}".format(generated_tokens))

    generated_words = tokenizer.decode(generated_tokens[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("Output text: {}".format(generated_words))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--input", type=str, default="")
    args = parser.parse_args()
    main(args)
