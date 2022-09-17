# set current path to the parent, to enable absolute imports
import os
from pathlib import Path
curr_path = Path(os.getcwd()).parent
os.chdir(curr_path)

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import torch
import seaborn as sns
import sqlite3
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import random
import requests
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam
from torch.nn import MSELoss
import math
from torch.utils.data import SequentialSampler
from transformers import BertForSequenceClassification
from main.active_learning.utils import seed_pool_split, experiment_AL
from main.active_learning.datasets import ALDataset
from main.active_learning.plotting import plot_al_results

# put device onto GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
if torch.cuda.is_available():
  print(f"GPU name: {torch.cuda.get_device_name()}")

# read in: first 200 labeled instances
sent_df = pd.read_csv('files/datasets/labeled/l01_reuters_sample200.csv')

# read in: next 800 labeled instances and join
### comment this out if only wanting to run the first iteration of the experiment ###
# send_df2 = pd.read_csv('files/datasets/labeled/l02_reuters_sample800.csv')
# sent_df = pd.concat([sent_df, send_df2], axis='rows',ignore_index=True)

# drop the miscellaneous instances
sent_df = sent_df[sent_df['is_miscellaneous'] == False]
print(f'Total: {len(sent_df)} instances')

# extract sentences and valence/arousal labels as numpy arrays
sentences = sent_df.sentence.values
v_labels = sent_df.valence.values
a_labels = sent_df.arousal.values

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# measure the maximum sentence length
# this is needed for adjusting the BERT size later
max_len = 0
# For every sentence...
for sent in sentences:
    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))
print('Max sentence length: ', max_len)


# Tokenize all of the sentences and map the tokens to their word IDs.
input_ids = []
attention_masks = []
# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=70,  # Pad & truncate all sentences.
        padding='max_length',
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )

    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
v_labels = torch.tensor(v_labels)
a_labels = torch.tensor(a_labels)

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])
print('Valence: ', v_labels[0])
print('Arousal: ', a_labels[0])

# set all the random seeds
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)

# experiment parameters
TRAIN_SIZE = 0.75
methods = ['random', 'farthest-first', 'mc-dropout']
batch_sizes = [16, 32, 64]  # , 32, 64
lrs = [1e-5, 1e-6, 1e-7]  #

# initiate results storage for valence and arousal models
v_results, a_results = {}, {}

### EXPERIMENT MAIN LOOP ###
# loop over all batch_sizes
for batch_size in batch_sizes:

    # initiate storage in dictionary, in a nested manner
    v_results[batch_size], a_results[batch_size] = {}, {}

    # loop over all the learning rates
    for lr in lrs:

        # initiate storage in dictionary, in a nested manner
        v_results[batch_size][lr], a_results[batch_size][lr] = {}, {}

        # loop over all the methods
        for method in methods:
            # initiate storage in dictionary, in a nested manner
            v_results[batch_size][lr][method], a_results[batch_size][lr][method] = {}, {}

            # print out modifications of this experiment loop
            print("--" * 20)
            print(f"METHOD: {method}. BATCH SIZE: {batch_size}. LEARNING RATE: {lr} ")
            print("--" * 20)

            # seperate 25% for testing, 75% training
            v_train_ds, v_test_ds = seed_pool_split(input_ids, attention_masks, v_labels, seed_size=TRAIN_SIZE,
                                                    random_state=RANDOM_STATE)
            a_train_ds, a_test_ds = seed_pool_split(input_ids, attention_masks, a_labels, seed_size=TRAIN_SIZE,
                                                    random_state=RANDOM_STATE)

            # initiate seed and pool
            v_seed, v_pool = seed_pool_split(v_train_ds[0], v_train_ds[1], v_train_ds[2], seed_size=batch_size,
                                             random_state=RANDOM_STATE)
            a_seed, a_pool = seed_pool_split(a_train_ds[0], a_train_ds[1], a_train_ds[2], seed_size=batch_size,
                                             random_state=RANDOM_STATE)

            # v_results[method], a_results[method] = {}, {}
            # v_results[method][batch_size], a_results[method][batch_size] = {}, {}

            # download valence and arousal base models
            print("Downloading Valence model...")
            torch.manual_seed(42)
            v_model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=1,
                output_attentions=False,
                output_hidden_states=True)

            print("Downloading Arousal model...")
            torch.manual_seed(42)
            a_model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=1,
                output_attentions=False,
                output_hidden_states=True)

            # initialize Active Learning datasets
            v_seed_ds, v_pool_ds = ALDataset(v_seed[0], v_seed[1], v_seed[2]), ALDataset(v_pool[0], v_pool[1],
                                                                                         v_pool[2])

            a_seed_ds, a_pool_ds = ALDataset(a_seed[0], a_seed[1], a_seed[2]), ALDataset(a_pool[0], a_pool[1],
                                                                                         a_pool[2])

            # take a subsample only
            ### COMMENT THIS OUT TO TAKE WHOLE SAMPLE INTO CONSIDERATION ###
            # SAMPLE_SIZE = 12
            # RANDOM_SEED = 42
            # v_pool_ds = v_pool_ds.subsample(SAMPLE_SIZE, RANDOM_SEED)
            # a_pool_ds = a_pool_ds.subsample(SAMPLE_SIZE, RANDOM_SEED)

            print("Valence Sample Pool Size: ", len(v_pool_ds), "Arousal Sample Pool Size: ", len(a_pool_ds))

            # initate testsets
            v_test_set = TensorDataset(v_test_ds[0], v_test_ds[1], v_test_ds[2])
            a_test_set = TensorDataset(a_test_ds[0], a_test_ds[1], a_test_ds[2])

            # start the experiments
            print("--" * 20)
            print("RUNNING VALENCE EXPERIMENT")
            print("--" * 20)
            v_train_rmse_curve, v_test_loss_curve = experiment_AL(v_seed_ds, v_pool_ds, v_test_set, v_model, method, lr,
                                                                  batch_size, device)

            print("--" * 20)
            print("RUNNING AROUSAL EXPERIMENT")
            print("--" * 20)
            a_train_rmse_curve, a_test_loss_curve = experiment_AL(a_seed_ds, a_pool_ds, a_test_set, a_model, method, lr,
                                                                  batch_size, device)

            # store results
            v_results[batch_size][lr][method]['train'], v_results[batch_size][lr][method][
                'test'] = v_train_rmse_curve, v_test_loss_curve
            a_results[batch_size][lr][method]['train'], a_results[batch_size][lr][method][
                'test'] = a_train_rmse_curve, a_test_loss_curve

# produce graph
for batch_size in batch_sizes:
    for lr in lrs:
        plot_al_results(v_results, a_results, batch_size, lr)

# function to structure the results
v_res_summ = {}
v_res_summ['random'] = []
v_res_summ['farthest-first'] = []
v_res_summ['mc-dropout'] = []

# display valence results
for key in v_results.keys():
    for subkey in v_results[key].keys():
        for method in v_results[key][subkey].keys():
            v_res_summ[method].append(np.mean(v_results[key][subkey][method]['test']))


# display arousal results
a_res_summ = {}
a_res_summ['random'] = []
a_res_summ['farthest-first'] = []
a_res_summ['mc-dropout'] = []

for key in a_results.keys():
    for subkey in a_results[key].keys():
        for method in a_results[key][subkey].keys():
            a_res_summ[method].append(np.mean(a_results[key][subkey][method]['test']))


# make a table to summarize the results
idx_arrs = [
    [16, 16, 16, 32, 32, 32, 64, 64, 64],
    [1e-5, 1e-6, 1e-7, 1e-5, 1e-6, 1e-7, 1e-5, 1e-6, 1e-7]
    ]

idx_tuples = list(zip(*idx_arrs))

index = pd.MultiIndex.from_tuples(idx_tuples, names=["batch_size", "learning_rate"])

# construct the results dataframes
v_res_summ_df = pd.DataFrame(v_res_summ, index=index)
a_res_summ_df = pd.DataFrame(a_res_summ, index=index)

print('-------- Valence Results --------')
print(v_res_summ_df.to_latex())

print('-------- Arousal Results --------')
print(a_res_summ_df.to_latex())