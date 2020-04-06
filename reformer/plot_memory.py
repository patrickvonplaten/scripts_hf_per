#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=10)

x = np.asarray([2**10, 2**11, 2**12, 2**13, 2**14, 2**15])

y_bert_1 = np.asarray([24, 72, 284, 1020, 4059, 16160])
y_bert_8 = np.asarray([148, 592, 2180, 8297, 32600, 128000])

y_ref_1_2 = np.asarray([30, 74, 122, 190, 492, 1227])
y_ref_8_2 = np.asarray([156, 364, 806, 1697, 3791, 9691])

y_ref_1_4 = np.asarray([48, 124, 186, 368, 920, 2300])
y_ref_8_4 = np.asarray([324, 692, 1426, 3117, 6971, 18289])

y_16_gb = np.asarray([16000, 16000, 16000, 16000, 16000, 16000])


plt.xlim(min(x), max(x))
plt.ylim(0, 20000)

plt.scatter(x, y_ref_8_2, c='b', label="Reformer Layer 2 hashes")
plt.plot(x, y_ref_8_2, 'k--')

plt.scatter(x, y_ref_8_4, c='g', label="Reformer Layer 4 hashes")
plt.plot(x, y_ref_8_4, 'k--')

plt.scatter(x, y_bert_8, c='black', label="Bert Layer")
plt.plot(x, y_bert_8, 'k--')

plt.plot(x, y_16_gb, 'r--', label="16 Giga Bytes")

plt.title("Memory usage Bert Layer vs. Reformer Layer for Batch size=8")
plt.ylabel("Memory usage in Mega Bytes")
plt.xlabel("Sequence Length")

plt.legend()
plt.show()
