#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

ax.set_xscale('log', basex=2)
ax.set_yscale('log', basey=10)

x = np.asarray([2**10, 2**11, 2**12, 2**13, 2**14, 2**15])

#y_bert_1 = np.asarray([1, 1, 1, 5, 24, 116])
#y_ref_1_2 = np.asarray([2, 2, 2, 3, 5, 13])
#y_ref_1_4 = np.asarray([2, 2, 3, 5, 10, 23])

y_bert_8 = np.asarray([1, 3, 10, 44, 176, 700])
y_ref_8_2 = np.asarray([3, 4, 9, 17, 39, 94])
y_ref_8_4 = np.asarray([4, 8, 15, 32, 70, 175])

plt.xlim(min(x), max(x))
plt.ylim(0, 200)

#plt.scatter(x, y_ref_1_2, c='b', label="Reformer Layer 2 hashes")
#plt.plot(x, y_ref_1_2, 'k--')
#plt.scatter(x, y_ref_1_4, c='g', label="Reformer Layer 4 hashes")
#plt.plot(x, y_ref_1_4, 'k--')
#plt.scatter(x, y_bert_1, c='black', label="Bert Layer")
#plt.plot(x, y_bert_1, 'k--')
#plt.title("Time usage Bert Layer vs. Reformer Layer for Batch size=1")

plt.scatter(x, y_ref_8_2, c='b', label="Reformer Layer 2 hashes")
plt.plot(x, y_ref_8_2, 'k--')
plt.scatter(x, y_ref_8_4, c='g', label="Reformer Layer 4 hashes")
plt.plot(x, y_ref_8_4, 'k--')
plt.scatter(x, y_bert_8, c='black', label="Bert Layer")
plt.plot(x, y_bert_8, 'k--')
plt.title("Time usage Bert Layer vs. Reformer Layer for Batch size=8")

plt.ylabel("Time usage in Milliseconds")
plt.xlabel("Sequence Length")

plt.legend()
plt.show()
