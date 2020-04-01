#!/usr/bin/env python3
from transformers import (
    T5ForConditionalGeneration,
    TFT5ForConditionalGeneration,
    T5Tokenizer,
)
from argparse import ArgumentParser  # noqa: F401


def main(args):
    size = args.size
    input_string = args.input

    tok = T5Tokenizer.from_pretrained(size)
    pt_input = tok.encode(input_string, return_tensors="pt")
    tf_input = tok.encode(input_string, return_tensors="tf")

    pt_model = T5ForConditionalGeneration.from_pretrained(size)
    tf_model = TFT5ForConditionalGeneration.from_pretrained(size)

    pt_output = pt_model.generate(
        pt_input,
        max_length=args.max_length,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        length_penalty=args.length_penalty,
        early_stopping=args.early_stopping,
    )
    tf_output = tf_model.generate(
        tf_input,
        max_length=args.max_length,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        length_penalty=args.length_penalty,
        early_stopping=args.early_stopping,
    )

    print('Equal: {}'.format(pt_output[0].numpy() == tf_output[0].numpy()))
    print('PT:')
    print(tok.decode(pt_output[0]))
    print(pt_output[0])
    print('TF:')
    print(tok.decode(tf_output[0]))
    print(tf_output[0])


if __name__ == "__main__":
    default_input = 'translate English to German: "Luigi often said to me that he never wanted the brothers to end up in court", she wrote.'

    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default=default_input)
    parser.add_argument("--size", type=str, default='t5-small')
    parser.add_argument("--max_length", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--length_penalty", type=float, default=2.0)
    parser.add_argument("--early_stopping", type=bool, default=True)
    args = parser.parse_args()
    main(args)
