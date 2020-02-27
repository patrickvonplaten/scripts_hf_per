#!/usr/bin/env python3
import torch
from torch.nn import functional as F
import tensorflow as tf
import numpy as np
import ipdb  # noqa: F401


def get_tensor_from_torch(path_to_vector):
    return torch.load(path_to_vector).detach().numpy()


def to_pt(np_tensor):
    return torch.from_numpy(np_tensor)


def to_tf(np_tensor):
    return tf.convert_to_tensor(np_tensor)


def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def tf_top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.shape[-1])  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < tf.math.top_k(logits, k=top_k)[0][..., -1, None]
        filter_value_logits = tf.Variable(tf.zeros_like(logits) + filter_value)
        logits = tf.where(indices_to_remove, filter_value_logits, logits)

    if top_p < 1.0:
        sorted_indices = tf.argsort(logits, direction="DESCENDING")
        sorted_logits = tf.gather(
            logits, sorted_indices, axis=-1, batch_dims=1
        )  # expects logits to be of dim (batch_size, vocab_size)

        cumulative_probs = tf.math.cumsum(
            tf.nn.softmax(sorted_logits, axis=-1), axis=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove = tf.concat(
                [
                    tf.zeros_like(sorted_indices_to_remove[:, :min_tokens_to_keep]),
                    sorted_indices_to_remove[:, min_tokens_to_keep:],
                ],
                -1,
            )
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove = tf.roll(sorted_indices_to_remove, 1, axis=-1)
        sorted_indices_to_remove = tf.concat(
            [
                tf.zeros_like(sorted_indices_to_remove[:, :1]),
                sorted_indices_to_remove[:, 1:],
            ],
            -1,
        )
        # scatter sorted tensors to original indexing
        batch_dim_indices_flat = tf.reshape(
            tf.broadcast_to(
                tf.expand_dims(tf.range(sorted_indices_to_remove.shape[0]), axis=-1),
                sorted_indices_to_remove.shape,
            ),
            [1, -1],
        )
        pair_sorted_indices_flat = tf.transpose(
            tf.concat([batch_dim_indices_flat, tf.reshape(sorted_indices, [1, -1])], 0)
        )
        indices_to_remove = tf.scatter_nd(
            pair_sorted_indices_flat,
            tf.reshape(sorted_indices_to_remove, [-1]),
            sorted_indices_to_remove.shape,
        )
        filter_value_logits = tf.Variable(tf.zeros_like(logits) + filter_value)
        logits = tf.where(indices_to_remove, filter_value_logits, logits)
    return logits


np_tensor = get_tensor_from_torch(
    "/home/patrick/hugging_face/scripts_per/tf_generate/logit_vector_sample.pt"
)[0]
np_tensor = np.tile(np_tensor, [2, 1])
tf_tensor = to_tf(np_tensor)
pt_tensor = to_pt(np.copy(np_tensor))

ipdb.set_trace()
out_pt = top_k_top_p_filtering(
    pt_tensor, top_k=50, top_p=0.7, min_tokens_to_keep=70
).numpy()
out_tf = tf_top_k_top_p_filtering(
    tf_tensor, top_k=50, top_p=0.7, min_tokens_to_keep=70
).numpy()

assert np.allclose(out_pt, out_tf)

print("yes")
