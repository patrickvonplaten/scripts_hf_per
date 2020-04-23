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
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.05,
        "seed": None  # that parameter is only needed for testing and will be removed soon
    })


def get_training_args():
    # define the training args
    return TrainingArguments(**{
        "learning_rate": 5e-4,
        "warmup_steps": 50,
        "adam_epsilon": 1e-9,
        "num_train_epochs": 600,
        "output_dir": "./runs",
        "do_train": True,
        "do_eval": True,
        "do_predict": True,
        "logging_steps": 10,
        "save_steps": 100,
    })


def prepare_dataset(max_length):
    # get pretrained tokenizer
    tokenizer = ReformerTokenizer.from_pretrained(
        "patrickvonplaten/reformer-crime-and-punish"
    )

    # define our map function to reduce the dataset to one sample
    def flatten_and_tokenize(batch):
        all_input_text = ["".join(batch["paragraph"])]
        input_ids_dict = tokenizer.batch_encode_plus(
            all_input_text, pad_to_max_length=True, max_length=max_length
        )
        return input_ids_dict

    # load the dataset
    dataset = nlp.load("crime_and_punish", split="train")

    # reduce the dataset
    dataset = dataset.map(
        flatten_and_tokenize, batched=True, remove_columns=["paragraph"]
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

        # return dict
        return {
            "input_ids": rolled_input_ids,  # BS x SEQ_LEN
            "lm_labels": rolled_input_ids,  # BS x SEQ_LEN
            "attention_mask": rolled_attention_mask,  # BS x SEQ_LEN
        }


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

    import ipdb
    ipdb.set_trace()

    # create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        eval_dataset=dataset,
        prediction_loss_only=True,
    )

    # train
    trainer.train()


if __name__ == "__main__":
    main()
