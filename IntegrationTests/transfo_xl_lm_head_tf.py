#!/usr/bin/env python3

from transformers import TFLogUniformSampler, LogUniformSampler
import torch
import tensorflow as tf

vocab_size = 10
num_sample = 3
labels = [1, 2]

samp_tf = TFLogUniformSampler(vocab_size, num_sample)
samp_pt = LogUniformSampler(vocab_size, num_sample)

print(samp_pt.sample(torch.tensor(labels)))
print(samp_tf.sample(tf.convert_to_tensor(labels)))
