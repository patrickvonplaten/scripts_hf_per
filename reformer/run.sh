#!/usr/bin/env bash

./benchmark_reformer_bert.py --verbose --save_to_csv --batch_sizes 1 8 --slice_sizes 1024 2048 4096 8192 16384 32768 --torch --torch_cuda --num_hashes 4 > reformer_line_by_line_num_hashes_4_fac_buckets.txt
