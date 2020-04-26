#!/usr/bin/env python3
from transformers import (
    ReformerModelWithLMHead,
    ReformerTokenizer,
    ReformerConfig,
    Trainer,
    DataCollator,
    TrainingArguments,
)
import nlp
import torch
import numpy as np


import logging
logging.basicConfig(level=logging.INFO)


def create_reformer_config():
    # define config of reformer model
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
        "lsh_attention_probs_dropout_prob": 0.0,
        "lsh_num_chunks_before": 1,
        "lsh_num_chunks_after": 0,
        "local_num_chunks_before": 1,
        "local_num_chunks_after": 0,
        "local_attention_probs_dropout_prob": 0.05,
        "hidden_dropout_prob": 0.05,
        "seed": None  # that parameter is only needed for testing and will be removed soon
    })


def get_training_args():
    # define the training args
    return TrainingArguments(**{
        "learning_rate": 0.01,
        "max_steps": 10000,
        "output_dir": "./output_2",
        "logging_dir": "./log_2",
        "do_train": True,
        "do_eval": True,
        "evaluate_during_training": True,
        "gradient_accumulation_steps": 8,
        "logging_steps": 50,
        "scheduler": "cosine_decay_hard_restarts",
        "num_cycles_cosine_decay": 0.7,
        "warmup_steps": 800,
        "weight_decay": 0.0,
        "adam_beta_1": 0.86,
        "adam_beta_2": 0.92,
        "adam_epsilon": 1e-9,
        "save_steps": 1000,
        "overwrite_output_dir": True
    })


def prepare_dataset(max_length):
    # get pretrained tokenizer
    tokenizer = ReformerTokenizer.from_pretrained(
        "patrickvonplaten/reformer-crime-and-punish"
    )

    # define our map function to reduce the dataset to one sample
    def flatten_and_tokenize(batch):
        all_input_text = ["".join(batch["line"])]
        input_ids_dict = tokenizer.batch_encode_plus(
            all_input_text, pad_to_max_length=True, max_length=max_length
        )

        return input_ids_dict

    # load the dataset
    dataset = nlp.load("crime_and_punish", split="train")

    # reduce the dataset
    dataset = dataset.map(
        flatten_and_tokenize, batched=True, batch_size=-1, remove_columns=["line"]
    )

    # prepare dataset to be in torch format
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return dataset


class ReformerCollator(DataCollator):
    def __init__(self, max_roll_length):
        self.max_roll_length = max_roll_length

    # From the official notebook: "Normally we would have a dataset with many examples, but for this demonstration we fit a language model on the single novel only. We don't want the model to just memorize the dataset by encoding the words in its position embeddings, so at each training iteration we will randomly select how much padding to put before the text vs. after it"
    def collate_batch(self, features):
        # get random shift int
        random_shift_length = torch.randint(self.max_roll_length, (1,)).item()

        # shift input and mask
        rolled_input_ids = torch.roll(
            features[0]["input_ids"], random_shift_length
        ).unsqueeze(0)
        rolled_attention_mask = torch.roll(
            features[0]["attention_mask"], random_shift_length
        ).unsqueeze(0)

        # set lm_labels = -100 for padded input
        lm_labels = torch.where(rolled_attention_mask == 0, torch.tensor(-100), rolled_input_ids)

        # return dict
        return {
            "input_ids": rolled_input_ids,  # BS x SEQ_LEN
            "labels": lm_labels,  # BS x SEQ_LEN
            "attention_mask": rolled_attention_mask,  # BS x SEQ_LEN
        }


def compute_metrics(pred):
    arg_max = np.argmax(pred.predictions, axis=-1)
    acc = np.mean(np.asarray((arg_max == pred.label_ids), dtype=np.float))
    return {"accuracy": acc}


def main():
    # let's use > 0.5M samples per sample
    padded_sequence_length = 2 ** 19

    # reduce dataset to one example
    dataset = prepare_dataset(padded_sequence_length)

    # the non_padded_sequence_length defines the max shift for our data collator
    non_padded_sequence_length = padded_sequence_length - sum(
        dataset["attention_mask"][0]
    )

    # use a special data collator that randomely shifts the input_ids
    data_collator = ReformerCollator(non_padded_sequence_length)

    # create reformer config and init model
    config = create_reformer_config()
    model = ReformerModelWithLMHead(config)

    # create training params
    training_args = get_training_args()

    # create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    # train
    trainer.train()


if __name__ == "__main__":
    main()
