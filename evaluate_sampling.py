import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from networking.common_lingo import ATTACK_TO_STRING, Attack
from networking.readers import NetCSVReader
from networking.samplers import WeightedSampler
from sampleddetection.utils import to_canadian

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path_str", default="./data/Wednesday.csv")
parser.add_argument("--sampling_budget", default=12, type=int)
parser.add_argument("--weighted_bins_num", default=1600, type=int)
parser.add_argument("--no-autoindent", action="store_true")

%load_ext autoreload # type: ignore
%autoreload 2# type: ignore

args = parser.parse_args()

csv_path = Path(args.csv_path_str)
csv_reader = NetCSVReader(csv_path)

# Find first SLOWLORIS inside the csv
slowloris_idx = csv_reader.csv_df["label"].str.contains("DoS_Slowloris")
slowloris_idx_min = slowloris_idx.idxmax()
slowlors_time_min = csv_reader.csv_df["timestamp"].iloc[slowloris_idx_min]
# Find last SLOWLORIS inside the csv
slowloris_idx = slowloris_idx[::-1]
slowloris_idx_max = slowloris_idx.idxmax()
slowlors_time_max = csv_reader.csv_df["timestamp"].iloc[slowloris_idx_max]

labels = [
    Attack.BENIGN,
    Attack.HULK,
    Attack.GOLDENEYE,
    Attack.SLOWLORIS,
    Attack.SLOWHTTPTEST,
]
lval_to_str = {l.value: ATTACK_TO_STRING[l] for l in labels}

global_sampler = WeightedSampler(
    csv_reader, args.sampling_budget, args.weighted_bins_num, labels
)
samples_count_per_label = {
    l.value: 0 for l in labels
}

# % Start creating some samples and evaluating what the packets ahead look like
sample, all_samples = global_sampler.sample_debug(0, 1e-2, 1e-3, first_sample=True)
print(f"Lengths for samples:{len(sample)} and all_samples:{len(all_samples)}")
samples_times = np.array([s.time for s in sample])
samples_labels = np.array([s.label.value for s in sample])
# Add the counts for each label
for s in sample:
    samples_count_per_label[s.label.value] += 1
print(f"Samples labels are {samples_labels}")
print(f"Samples labels are {samples_count_per_label}")

# Create a bar plot of the samples count
plt.figure(figsize=(8, 6))
plt.bar(list(lval_to_str.keys()), list(samples_count_per_label.values()))
plt.xticks(list(lval_to_str.keys()), list(lval_to_str.values()))
plt.xlabel("Label")
plt.ylabel("Samples Count")
plt.title("Samples Count per Label (Weighted Time Sampling)")
for i, v in enumerate(list(samples_count_per_label.values())):
    plt.text(i, v + 1, f"{v}", ha="center", va="bottom")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(15, 5))

colors = ["r", "g", "b", "c", "m", "y", "k"]  # Add more colors if
# Simply plot scatter plots with different colors for each label
for l in labels:
    idxs = np.where(samples_labels == l.value)[0]
    label_times = samples_times[idxs]
    # Each label with a different color
    color = colors[l.value]
    ax.scatter(
        label_times[idxs],
        np.full_like(label_times, 1),
        label=ATTACK_TO_STRING[l],
        marker=color,
        s=100,
    )

# For later
# arraylike_features, labels = self.feature_factory.make_feature_and_label(new_samples)
