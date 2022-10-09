"""
Experiment to investigate how much batching speeds up the learning process

NOTE: we only do it on valence dimension here
"""

# set current path to the parent, to enable absolute imports
import os
from pathlib import Path
curr_path = Path(os.getcwd()).parent
os.chdir(curr_path)

# imports
import torch
from torch.utils.data import TensorDataset
import pandas as pd
from transformers import BertTokenizer
from main.active_learning.utils import tokenize_sentences, set_device
import random
import numpy as np
from main.active_learning.utils import experiment_AL, init_dataloader, seed_pool_split
from main.active_learning.datasets import ALDataset
from transformers import BertForSequenceClassification
import time

# set device (cpu or gpu)
DEVICE = set_device()

# read in data
sent_df = pd.read_csv('files/datasets/labeled/l01_reuters_sample200.csv')
sent_df = sent_df[sent_df['is_miscellaneous'] == False]
print(f'Total: {len(sent_df)} instances')

# extract sentences and valence/arousal labels as numpy arrays
sentences = sent_df.sentence.values
v_labels = sent_df.valence.values

# handle tokenization of sentences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
MAX_LEN = 70
input_ids, attention_masks = tokenize_sentences(tokenizer, sentences, MAX_LEN)

# convert label arrays into tensors
v_labels = torch.tensor(v_labels)

# set all the random seeds for reproducibility
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)


# fixed parameters + batch_sizes we want to investigate
TRAIN_SIZE = 0.99
EPOCHS = 1
LR = 1e-05
batch_sizes = [8, 16, 32, 64]
#batch_sizes = [64]
methods = ['random', 'farthest-first', 'mc-dropout']


# storage for results
res = {}


for batch_size in batch_sizes:

    # make entry in results dict
    res[batch_size] = {}

    for method in methods:

        # seperate 100% training, 0% testing
        v_train_ds, v_test_ds = seed_pool_split(input_ids, attention_masks, v_labels, seed_size=TRAIN_SIZE,
                                                random_state=RANDOM_STATE)

        # print how many training instances we are using
        print(f"Utilizing {len(v_train_ds[2])} training instances")
        print(f"Utilizing {len(v_test_ds[2])} testing instances")

        # initiate seed and pool
        v_seed, v_pool = seed_pool_split(v_train_ds[0], v_train_ds[1], v_train_ds[2], seed_size=batch_size,
                                         random_state=RANDOM_STATE)

        # initialize new model
        torch.manual_seed(42)
        v_model = BertForSequenceClassification.from_pretrained(
                   'bert-base-uncased',
                   num_labels=1,
                   output_attentions=False,
                   output_hidden_states=True)

        # initialize new AL Dataset
        v_seed_ds, v_pool_ds = ALDataset(v_seed[0], v_seed[1], v_seed[2]), ALDataset(v_pool[0], v_pool[1],
                                                                                     v_pool[2])

        # initialize (empty) testset
        v_test_set = TensorDataset(v_test_ds[0], v_test_ds[1], v_test_ds[2])

        print(f"Measuring time for batch size {batch_size} and method {method}")

        # start training function
        start_time = time.time()
        train_rmse_curve, test_rmse_curve = experiment_AL(v_seed_ds, v_pool_ds, v_test_set, v_model, method, LR, batch_size, DEVICE)
        end_time = time.time()

        # measure elapsed time and store it
        elapsed_time = end_time - start_time
        res[batch_size][method] = elapsed_time

        # print out result
        print(f"Computation time for batch size {batch_size} and method {method}: {elapsed_time:.4f}")

        #  write to file
        file_path = os.getcwd() + "\\files\\results\\active_learning_experiments\\logs\\computation_times.txt"

        with open(file_path, "a+") as v_file:
            v_file.write(f"batch_size: {batch_size}, method: {method}, computation_time: {elapsed_time:.4f}" + "\n")


# create a nice latex table and store it
res_table = pd.DataFrame(res)

table_file_path = os.getcwd() + "\\files\\results\\tables\\computation_time_table.txt"
with open(table_file_path, "w+") as v_file:
    v_file.write(res_table.to_latex())









