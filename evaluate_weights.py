"""
Simple to do some analysis on the weights
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Import the data
import numpy as np

from networking.common_lingo import ATTACK_TO_STRING, STRING_TO_ATTACKS, Attack

bin_times, bin_labels, bin_weights = np.load("./.cache_dir/bins.npy")
bin_labels = bin_labels.astype(np.int8)

labels = [
    Attack.BENIGN,
    Attack.HULK,
    Attack.GOLDENEYE,
    Attack.SLOWLORIS,
    Attack.SLOWHTTPTEST,
]


plt.figure(figsize=(8, 19))
plt.tight_layout()

lval = np.array([l.value for l in labels])

colors = ["r", "g", "b", "c", "m", "y", "k"]  # Add more colors if
for l in labels:
    idxs = np.where(bin_labels == l.value)[0]
    # Each label with a different color
    color = colors[l.value]
    plt.stem(
        bin_times[idxs], bin_weights[idxs], label=ATTACK_TO_STRING[l], markerfmt=color
    )
plt.xlabel("Time")
plt.ylabel("Weight")
plt.legend()
plt.show()
