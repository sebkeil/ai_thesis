"""
File to run the experiment comparing joint and disjoint models by measuring RMSE error metric.
"""

# set the path to enable relative imports
import os
from pathlib import Path
curr_path = Path(os.getcwd()).parent
os.chdir(curr_path)

# import relevant libraries
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd
import numpy as np
from main.active_learning.datasets import ALDataset
from main.active_learning.utils import seed_pool_split, init_dataloader
from main.joint_disjoint.utils import train_joint, train_disjoint, test_joint, test_disjoint
from main.active_learning.utils import tokenize_sentences, set_device


# set device (cpu or gpu)
DEVICE = set_device()

# read in data
sent_df = pd.read_csv('files/datasets/labeled/l01_reuters_sample200.csv')
sent_df = sent_df[sent_df['is_miscellaneous'] == False]
print(f'Total: {len(sent_df)} instances')

#toDO: extend this to full 1000 instances


# extract sentences and valence/arousal labels as numpy arrays
sentences = sent_df.sentence.values
v_labels = sent_df.valence.values
a_labels = sent_df.arousal.values
va_labels = np.vstack((v_labels, a_labels)).T

# handle tokenization of sentences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_LEN = 70
input_ids, attention_masks = tokenize_sentences(tokenizer, sentences, MAX_LEN)

# convert label arrays into tensors
v_labels = torch.tensor(v_labels)
a_labels = torch.tensor(a_labels)
va_labels = torch.tensor(va_labels)


### MODEL DEFINITION ###

models = {}

# define joint model
models['va_joint'] = BertForSequenceClassification.from_pretrained(
             'bert-base-uncased',
             num_labels=2,
             output_attentions=False,
             output_hidden_states=True)

# define the two disjoint models
models['v_disjoint'] = BertForSequenceClassification.from_pretrained(
               'bert-base-uncased',
               num_labels=1,
               output_attentions=False,
               output_hidden_states=True)

# define the two disjoint models
models['a_disjoint'] = BertForSequenceClassification.from_pretrained(
               'bert-base-uncased',
               num_labels=1,
               output_attentions=False,
               output_hidden_states=True)

# define experiment parameters
RANDOM_STATE = 42
TRAIN_SIZE = 0.75
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-5


# seperate 75% training set, 25% test set
va_train_ds, va_test_ds = seed_pool_split(input_ids, attention_masks, va_labels, seed_size=TRAIN_SIZE,
                                                    random_state=RANDOM_STATE)

v_train_ds, v_test_ds = seed_pool_split(input_ids, attention_masks, v_labels, seed_size=TRAIN_SIZE, random_state=RANDOM_STATE)
a_train_ds, a_test_ds = seed_pool_split(input_ids, attention_masks, a_labels, seed_size=TRAIN_SIZE, random_state=RANDOM_STATE)



# initialize AL Dataset for training and testing data
va_train_ds_al = ALDataset(va_train_ds[0], va_train_ds[1], va_train_ds[2])
va_test_ds_al = ALDataset(va_test_ds[0], va_test_ds[1], va_test_ds[2])

v_train_ds_al = ALDataset(v_train_ds[0], v_train_ds[1], v_train_ds[2])
v_test_ds_al = ALDataset(v_test_ds[0], v_test_ds[1], v_test_ds[2])

a_train_ds_al = ALDataset(a_train_ds[0], a_train_ds[1], a_train_ds[2])
a_test_ds_al = ALDataset(a_test_ds[0], a_test_ds[1], a_test_ds[2])


# initialize dataloaders
va_train_dl = init_dataloader(va_train_ds_al, BATCH_SIZE, type='random')
va_test_dl = init_dataloader(va_test_ds_al, BATCH_SIZE, type='random')

v_train_dl = init_dataloader(v_train_ds_al, BATCH_SIZE, type='random')
v_test_dl = init_dataloader(v_test_ds_al, BATCH_SIZE, type='random')

a_train_dl = init_dataloader(a_train_ds_al, BATCH_SIZE, type='random')
a_test_dl = init_dataloader(a_test_ds, BATCH_SIZE, type='random')


# start training
valence_rmse_curve_joint_train, arousal_rmse_curve_join_train = train_joint(va_model, va_dataloader, EPOCHS, LR, device)



