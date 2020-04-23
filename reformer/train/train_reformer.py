#!/usr/bin/env python3
from transformers import (
    ReformerModelWithLMHead,
    ReformerTokenizer,
    ReformerConfig,
    Trainer,
    DataCollator,
    HfArgumentParser,
    TrainingArguments,
    start_memory_tracing,
    stop_memory_tracing
)
import nlp
import torch


def print_summary_statistics(summary):
    print(
        "\nLines by line memory consumption:\n"
        + "\n".join(
            f"{state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}"
            for state in summary.sequential
        )
    )
    print(
        "\nLines with top memory consumption:\n"
        + "\n".join(
            f"=> {state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}"
            for state in summary.cumulative[:6]
        )
    )
    print(
        "\nLines with lowest memory consumption:\n"
        + "\n".join(
            f"=> {state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}"
            for state in summary.cumulative[-6:]
        )
    )
    print(f"\nTotal memory increase: {summary.total}")


def create_reformer_config():
    return ReformerConfig(**{
        "attention_head_size": 64,
        "attn_layers": ["local", "lsh", "local", "lsh", "local", "lsh"],
        "axial_pos_embds": True,
        "sinusoidal_pos_embds": False,
        "axial_pos_embds_dim": [64, 192],
        "axial_pos_shape": [512, 1024],
        "lsh_attn_chunk_length": 64,
        "local_attn_chunk_length": 64,
        "feed_forward_size": 512,
        "hidden_act": "relu",
        "hidden_size": 256,
        "is_decoder": True,
        "max_position_embeddings": 524288,
        "num_attention_heads": 2,
        "num_buckets": [64, 128],
        "num_hashes": 1,
        "vocab_size": 320,
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.05,
        "seed": None
    })


def prepare_dataset(max_length):
    tokenizer = ReformerTokenizer.from_pretrained(
        "patrickvonplaten/reformer-crime-and-punish"
    )

    def flatten_and_tokenize(batch):
        all_input_text = ["".join(batch["paragraph"])]
        input_ids_dict = tokenizer.batch_encode_plus(
            all_input_text, pad_to_max_length=True, max_length=max_length
        )
        return input_ids_dict

    dataset = nlp.load("crime_and_punish", split="train[:1%]")
    dataset = dataset.map(
        flatten_and_tokenize, batched=True, remove_columns=["paragraph"]
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return dataset


class ReformerCollator(DataCollator):
    def __init__(self, max_roll_length):
        self.max_roll_length = max_roll_length

    def collate_batch(self, features):
        # randomely shift the tokens here as is done in
        # https://colab.research.google.com/github/google/trax/blob/master/trax/models/reformer/text_generation.ipynb
        random_shift_length = torch.randint(self.max_roll_length, (1,)).item()
        rolled_input_ids = torch.roll(
            features[0]["input_ids"], random_shift_length
        ).unsqueeze(0)
        rolled_attention_mask = torch.roll(
            features[0]["attention_mask"], random_shift_length
        ).unsqueeze(0)

        return {
            "input_ids": rolled_input_ids,  # BS x SEQ_LEN
            "lm_labels": rolled_input_ids,  # BS x SEQ_LEN
            "attention_mask": rolled_attention_mask,  # BS x SEQ_LEN
        }


def main(training_args):

    padded_sequence_length = 2 ** 19
    dataset = prepare_dataset(padded_sequence_length)
    non_padded_sequence_length = padded_sequence_length - sum(
        dataset["attention_mask"][0]
    )
    data_collator = ReformerCollator(non_padded_sequence_length)
    config = create_reformer_config()

    model = ReformerModelWithLMHead(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=dataset,
        prediction_loss_only=True,
    )

    trace = start_memory_tracing("transformers")
    trainer.train()
    summary = stop_memory_tracing(trace)
    print_summary_statistics(summary)


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments))
    training_args = parser.parse_args_into_dataclasses()[0]
    main(training_args)
