#!/usr/bin/env python3

from transformers import LongformerTokenizer, LongformerForQuestionAnswering
import torch
import nlp


validation_dataset = nlp.load_dataset("trivia_qa", "rc", split="validation[:1%]")


def set_context(example):
    example["context"] = " ".join(("\n".join(example["entity_pages"]["wiki_context"])).split("\n"))
    example["targets"] = example["answer"]["aliases"]
    return example


# We only care about examples of the wikipedia part of the library
validation_dataset = validation_dataset.map(set_context, load_from_cache_file=False, remove_columns=["search_results", "question_source", "entity_pages", "answer"])

validation_dataset = validation_dataset.filter(lambda x: len(x["context"]) > 0)

validation_dataset.map(lambda x, i: print(f"Id: {i} - Length: {len(x['context'])}"), with_indices=True)

tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")

def evaluate(example):
    def get_answer(question, text):
        encoding = tokenizer.encode_plus(question, text, return_tensors="pt", max_length=tokenizer.max_len)

        # the forward method will automatically set global attention on question tokens
        start_scores, end_scores = model(**encoding)

        all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
        answer_tokens = all_tokens[torch.argmax(start_scores): torch.argmax(end_scores)+1]
        answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))[1:]  # remove space prepending space token
        return answer

    example["output"] = get_answer(example["question"], example["context"])
    example["match"] = (example["output"] in example["answers"])
    return example


result = evaluate(validation_dataset[0])

results = validation_dataset.map(evaluate, load_from_cache_file=False)
