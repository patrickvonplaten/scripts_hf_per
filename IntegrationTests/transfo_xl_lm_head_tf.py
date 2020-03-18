#!/usr/bin/env python3
from transformers import TFTransfoXLLMHeadModel, TransfoXLLMHeadModel
import torch
import tensorflow as tf


#tf_model = TFTransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103', sample_softmax=3)
model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103', sample_softmax=3)

tf_input = tf.convert_to_tensor([[0, 1]])
pt_input = torch.tensor([[0, 1]])

pt_output = model(pt_input, training=True)
#tf_output = tf_model(tf_input, training=True)
