# %% [markdown]
# # Introduction
# Point of this notebook is simply to observe (withotu any decision making) the performance across a matrix of window skips and window lengths

import os
import random
from itertools import product
from pathlib import Path
from typing import List

import numpy as np
from tqdm.notebook import tqdm

from sampleddetection.common_lingo import Attack
from sampleddetection.datastructures.flowsession import SampledFlowSession
# %%
from sampleddetection.environment.datastructures import Action, State
from sampleddetection.environment.model import Environment

# Make sure these are reloaded when cells are rerun
# %load_ext autoreload
# %autoreload 2

# %%
# Setup the environment
# From Microsecond to dekasecond
window_skips    = np.logspace(-6, 1, 1, dtype=float) # DEBUG: This is like this while we test lengths 
window_lengths  = np.logspace(-6, -3, 4, dtype=float)
#window_lengths  = 2*np.linspace(0.01, , 3, dtype=float)

# Reinforcement Learning Training parameters
batch_size      = 16
simultaneous_simulations = 4
agent_hidden_dim = 64

multiclass_classifcation = True # If false, we label all attacks as 1
csv_path = './data/Wednesday.csv'
dataset_dir    = './data/precalc_windows/'
dataset_filename = 'ws_{}_wl_{}.csv'
desired_features = [
            # Debugging info
            #"start_ts",
            #"start_timestamp",
            #"end_timestamp",
            #"tot_fwd_pkts",
            #"tot_bwd_pkts",
            # Non debugging
            "label",
            "fwd_pkt_len_max",
            "fwd_pkt_len_min",
            "fwd_pkt_len_mean",
            "bwd_pkt_len_max",
            "bwd_pkt_len_min",
            "bwd_pkt_len_mean",
            "flow_byts_s",
            "flow_pkts_s",
            "flow_iat_mean",
            "flow_iat_max",
            "flow_iat_min",
            "fwd_iat_mean",
            "fwd_iat_max",
            "fwd_iat_min",
            "bwd_iat_max",
            "bwd_iat_min",
            "bwd_iat_mean",
            "pkt_len_min",
            "pkt_len_max",
            "pkt_len_mean",
]
attacks_to_detect = [
    Attack.SLOWLORIS,
    Attack.SLOWHTTPTEST,
    Attack.HULK,
    Attack.GOLDENEYE,
    #Attack.HEARTBLEED. # Takes to long find in dataset.
]

# Use product to get a matrix of combinations
options_matrix = list(product(window_skips, window_lengths))
print(f"Working with {len(options_matrix)} permutaitions")

from joblib import load

from sampleddetection.common_lingo import ATTACK_TO_STRING, Attack
from sampleddetection.environment.models import Agent
from sampleddetection.readers.readers import CSVReader
# %%
# Create or Load dataset
from sampleddetection.samplers.window_sampler import NoReplacementSampler
from sampleddetection.writers.convenience import save_flows_to_csv

# TODO: Perhaps make it so the enviornment can tell use when we are sampling around the same areas. 
environments = [ Environment(sampler) for i in range(simultaneous_simulations)]
agent_input_dim = len(desired_features)
baseline_classifier = load("models/detection_model.joblib")
agent = Agent(agent_input_dim, agent_hidden_dim, 2, desired_features)

# At this point we might want to use MPI to create multiple simulations and cooridnate them

# %%
# Create the simulations
for i in range(simultaneous_simulations):



exit()
# Everything below this is not yet considered
from typing import Dict, Tuple

# %%
from sampleddetection.datastructures.flow import Flow


def generate_sessions(ws: float, wl: float) -> List[Tuple[Tuple,Flow]]:
    """Ensure we get a balanced sampling from the large dataset."""
    cur_amnt = 0
    flows: List[Tuple,Flow] = []
    if multiclass_classifcation:
        count_per_class = {attack: 0 for attack in attacks_to_detect}
        count_per_class[Attack.BENIGN] = 0
        total_amount = min_necessary_classes*(len(attacks_to_detect)+1)
        inner_bar = tqdm(total=total_amount,desc=f'Generating ws: {ws}- wl: {wl} flow',leave=False)
    else:
        count_per_class = {Attack.BENIGN: 0, Attack.GENERAL: 0}
        total_amount = min_necessary_classes*2
        inner_bar = tqdm(total=total_amount,desc=f'Generating ws: {ws}- wl: {wl} flow',leave=False)

    enough_samples_per_class = {class_name: False for class_name in count_per_class.keys()}
    sampler.clear_memory()
    while all(enough_samples_per_class.values()) == False:

        flow_sesh =  sampler.sample(winskip=ws,winlen=wl).flow_sesh
        # Count the distributions
        label_distributions = flow_sesh.flow_label_distribution()
        # For now just predict binary attack-benining
        for kflow, flow in flow_sesh.flows.items() : 
            if multiclass_classifcation:
                label = flow.label
            else:
                label = Attack.GENERAL if flow.label != Attack.BENIGN else Attack.BENIGN

            if label not in count_per_class or count_per_class[label] >= min_necessary_classes:
                continue # Dont over add

            count_per_class[label] += 1
            if count_per_class[label] == min_necessary_classes:
                enough_samples_per_class[label] = True
                
            flows.append((kflow,flow))

            inner_bar.update(1)
    return flows

# %%

flows = {}
# Set random seeds:
np.random.seed(0)
random.seed(0)
import csv

# Generate the datasets
for ws, wl in tqdm(options_matrix,desc='Creating datasets'):
    # Check if datasets exists
    flows = {f"ws:{ws}-ws:{wl}" : []}
    target_name = os.path.join(dataset_dir,dataset_filename.format(ws, wl))
    if os.path.exists(target_name):
        print(f"Will later be Loading {dataset_filename.format(ws, wl)} from {dataset_dir}")
        continue
    sesh = generate_sessions(ws,wl)

    ds_path = os.path.join(dataset_dir,dataset_filename.format(ws, wl))
    save_flows_to_csv(sesh, ds_path, desired_features=desired_features, samples_per_class=samples_per_class,overwrite=True, multiclass=multiclass_classifcation)
    
!notify-send done


# %%
# Show the sampling windows to ensure they are disjoint
import matplotlib.pyplot as plt

from sampleddetection.common_lingo import Attack
from sampleddetection.datastructures.flow import Flow

max_y = 100
y_increase = 100/len(flows)

time_windows = [f.time_window for f in flows]

# Ideally these two should be the same.
time_windows = sorted(time_windows, key=lambda tw: tw.start)
time_windows_end_wise = sorted(time_windows, key=lambda tw: tw.end)
    
# Actuallly plot them
for tw in time_windows:
    plt.plot([tw.start,tw.end],[i*y_increase,i(y_increase)],label='start', marker='o')
plt.xlabel('Time')
plt.ylabel('Time Windows')
plt.xlim(time_windows[0].start,1time_windows_end_wise[-1].end)
plt.title("Timeline of Windows")
plt.show()
    

# %%
# Load PreCalced datasets
for ws, wl in tqdm(options_matrix,desc='Loading datasets'):
    target_name = os.path.join(dataset_dir,dataset_filename.format(ws, wl))
    if not os.path.exists(target_name):
        print(f"Could not find {target_name}")
        raise FileNotFoundError

# %% [markdown]
# # Training Model on Different Schedules
# 
# We will use the matrix of different parameters to see how the training changes performance.

# %%
# This will be a function that will take flows calculated/loaded up above and will train the model. 
# It will return data of  the training and testing results to later be plotted in a loop that will call it
import pandas as pd
from sklearn.model_selection import train_test_split

from sampleddetection.util.data import (clean_dataset,
                                        train_classifier_XGBoost,
                                        train_multinary_classier_XGBoost)

features = [
            "label",
            "fwd_pkt_len_max",
            "fwd_pkt_len_min",
            "fwd_pkt_len_mean",
            "bwd_pkt_len_max",
            "bwd_pkt_len_min",
            "bwd_pkt_len_mean",
            "flow_byts_s",
            "flow_pkts_s",
            "flow_iat_mean",
            "flow_iat_max",
            "flow_iat_min",
            "fwd_iat_mean",
            "fwd_iat_max",
            "fwd_iat_min",
          #  "bwd_iat_max",
            "bwd_iat_min",
            "bwd_iat_mean",
            "pkt_len_min",
            "pkt_len_max",
            "pkt_len_mean",
]


def evaluate_performance(df: pd.DataFrame, ws: float, wl: float) -> Dict:
    """Evaluate the performance of the model with the given dataset."""
    # Clean the dataset
    df_ddos = clean_dataset(df,features, attacks_to_detect, (multiclass_classifcation == False))

    # Train the Model
    X_train, X_test, y_train, y_test = train_test_split(
        df_ddos.drop(columns=["label"]), df_ddos["label"], test_size=0.3
    )

    if multiclass_classifcation:
      mode, evals =  train_multinary_classier_XGBoost(X_train,  y_train,X_test, y_test)

    else:
      mode, evals = train_classifier_XGBoost(X_train,  y_train,X_test, y_test)

    return evals
    

# %%
# This will be the outer loop that will vall evaluete_performance
accuracies = []
log_losses = []
roc_aucs = []
for ws, wl in tqdm(options_matrix,desc='Evaluating datasets'):
    target_name = os.path.join(dataset_dir,dataset_filename.format(ws, wl))
    if not os.path.exists(target_name):
        print(f"Could not find {target_name}")
        raise FileNotFoundError
    # Load the data
    df = pd.read_csv(target_name)
    # Evaluate the data
    metrics = evaluate_performance(df, ws, wl)
    accuracies.append(metrics["accuracy"])
    log_losses.append(metrics["log_loss"])
    #roc_aucs.append(metrics["roc_auc"])

# Plot

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Simple Graph disregarding winskips

# Setup the figure
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

wl_notation = [f"{wl:.2e}" for wl in window_lengths]

# Plot dotted curve  with seaborn
sns.lineplot(x=wl_notation, y=accuracies, ax=ax[0], label="Accuracy")
sns.lineplot(x=wl_notation, y=log_losses, ax=ax[1], label="Log Loss")

# Set the labels
ax[0].set_xlabel("Window Length")
ax[0].set_ylabel("Accuracy")
ax[0].set_title("Accuracy vs Window Length")
ax[0].legend()




# %%
# Heat Map Plotting(FOR BINARY CLASSIFICATION ONLY FOR NOW)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, LogFormatter

#winskips = [f"{ws:.2f}" for ws, wl in options_matrix]
#winlens = [f"{wl:.2f}" for ws, wl in options_matrix]
winskips = [f"{i:.2e}" for i in np.logspace(-6, 1, 1)]
winlens = [f"{i:.2e}" for i in np.logspace(-3, 1, 4)]
# Format these with scientific (1e-6) notation

fig, ax = plt.subplots(3,1,figsize=(5,10))

#formatter = LogFormatter(10, labelOnlyBase=False)

print(winskips)

sns.heatmap(np.array(accuracies).reshape(1,4),ax=ax[0],annot=True,fmt=".2f", xticklabels=winskips, yticklabels=winlens)
ax[0].set_title("Accuracy")
ax[0].set_xlabel("Window Length")
ax[0].set_ylabel("Window Skip")

sns.heatmap(np.array(log_losses).reshape(1,4),ax=ax[1],annot=True,fmt=".2f",xticklabels=winskips, yticklabels=winlens)
ax[1].set_title("Log Loss")
ax[1].set_xlabel("Window Length")
ax[1].set_ylabel("Window Skip")

#sns.heatmap(np.array(roc_aucs).reshape(1,4),ax=ax[2],annot=True,fmt=".2f",xticklabels=winskips, yticklabels=winlens)
#ax[2].set_title("ROC AUC")
#ax[2].set_xlabel("Window Length")
#ax[2].set_ylabel("Window Skip")

plt.tight_layout()
plt.show()



