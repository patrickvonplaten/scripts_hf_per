#!/usr/bin/env python3
import t5
import tensorflow as tf
import mesh_tensorflow.transformer.dataset as transformer_dataset
import itertools

tf.enable_eager_execution()
task = t5.data.get_mixture_or_task("glue_cola_v002")
ds = task.get_dataset({"inputs": 64, "targets": 8}, "train")

ds = transformer_dataset.pack_or_pad(
    ds,
    {"inputs": 64, "targets": 8},
    pack=False,
    feature_keys=tuple(task.output_features),
    ensure_eos=True,
)


def add_attention_masks(ds, feature_keys):
    def _map_fn(ex):
        for key in feature_keys:
            tensor = ex[key]
            mask = tf.cast(tf.greater(tensor, 0), tensor.dtype)
            ex[key + "_mask"] = mask
        return ex
    return ds.map(
        _map_fn,
        num_parallel_calls=t5.data.preprocessors.num_parallel_calls()
    )


def add_decoder_inputs(ds, targets_key="targets"):
    def _map_fn(ex):
        targets = ex[targets_key]
        shifted_targets = tf.roll(targets, 1, 0)
        shifted_targets *= tf.cast(
            tf.logical_not(tf.equal(shifted_targets, 1)),
            shifted_targets.dtype,
        )
        ex["decoder_inputs"] = shifted_targets
        return ex
    return ds.map(
        _map_fn,
        num_parallel_calls=t5.data.preprocessors.num_parallel_calls()
    )


ds = add_attention_masks(ds, tuple(task.output_features))
ds = add_decoder_inputs(ds)

batch_size = 16  # @param {type:"integer"}
ds = ds.batch(batch_size)

for step, batch in enumerate(itertools.islice(ds, 2)):
    input_ids = batch["inputs"],
    attention_mask = batch["inputs_mask"]
    decoder_input_ids = batch["decoder_inputs"]
    decoder_attention_mask = batch["targets_mask"]
    lm_labels = batch["targets"]
    import ipdb
    ipdb.set_trace()
