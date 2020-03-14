#!/usr/bin/env python3

from transformers import (
    AutoTokenizer,
    AutoConfig,
    BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    BART_PRETRAINED_MODEL_ARCHIVE_MAP,
    OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP,
    TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP,
    GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
    CTRL_PRETRAINED_MODEL_ARCHIVE_MAP,
    XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
    XLM_PRETRAINED_MODEL_ARCHIVE_MAP,
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
    DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    T5_PRETRAINED_MODEL_ARCHIVE_MAP,
    FLAUBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
)

ALL_PRETRAINED_MODEL_ARCHIVE_MAP = dict(
    (key, value)
    for pretrained_map in [
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        BART_PRETRAINED_MODEL_ARCHIVE_MAP,
        OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP,
        TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_MAP,
        GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
        CTRL_PRETRAINED_MODEL_ARCHIVE_MAP,
        XLNET_PRETRAINED_MODEL_ARCHIVE_MAP,
        XLM_PRETRAINED_MODEL_ARCHIVE_MAP,
        ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
        DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        T5_PRETRAINED_MODEL_ARCHIVE_MAP,
        FLAUBERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
    ]
    for key, value, in pretrained_map.items()
)

for model_id_name in ALL_PRETRAINED_MODEL_ARCHIVE_MAP.keys():
    tok = AutoTokenizer.from_pretrained(model_id_name)
    conf = AutoConfig.from_pretrained(model_id_name)

    pad_equal = tok.pad_token_id == conf.pad_token_id
    eos_equal = tok.eos_token_id == conf.eos_token_id
    bos_equal = tok.bos_token_id == conf.bos_token_id

    if not pad_equal:
        print("PAD not equal for {}!".format(model_id_name))
        print("TOK: {} | CONF: {}".format(tok.pad_token_id, conf.pad_token_id))

    if not eos_equal:
        print("EOS not equal for {}!".format(model_id_name))
        print("TOK: {} | CONF: {}".format(tok.eos_token_id, conf.eos_token_id))

    if not bos_equal:
        print("BOS not equal for {}!".format(model_id_name))
        print("TOK: {} | CONF: {}".format(tok.bos_token_id, conf.bos_token_id))
