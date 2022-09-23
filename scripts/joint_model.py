
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
from main.active_learning.utils import seed_pool_split, init_dataloader, train


# put device onto GPU if available, else on CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
if torch.cuda.is_available():
  print(f"GPU name: {torch.cuda.get_device_name()}")


# read in the data
sent_df = pd.read_csv('files/datasets/labeled/l01_reuters_sample200.csv')
sent_df = sent_df[sent_df['is_miscellaneous'] == False]
print(f'Total: {len(sent_df)} instances')


# extract sentences and valence/arousal labels as numpy arrays
sentences = sent_df.sentence.values
v_labels = sent_df.valence.values
a_labels = sent_df.arousal.values
va_labels = np.vstack((v_labels, a_labels)).T

# tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize all of the sentences and map the tokens to their word IDs.
# toDO: abstract this away into a function
input_ids = []
attention_masks = []
# For every sentence...
for sent in sentences:
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
va_labels = torch.tensor(va_labels)


# define the joint model
va_model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=2,
                output_attentions=False,
                output_hidden_states=True)


# define dataset
RANDOM_STATE = 42
TRAIN_SIZE = 0.75
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-5


# seperate 75% training set, 25% test set
va_train_ds, va_test_ds = seed_pool_split(input_ids, attention_masks, va_labels, seed_size=TRAIN_SIZE,
                                                    random_state=RANDOM_STATE)


# NEW PART: initialize AL Dataset for training and testing data
va_train_ds_al = ALDataset(va_train_ds[0], va_train_ds[1], va_train_ds[2])
va_test_ds_al = ALDataset(va_test_ds[0], va_test_ds[1], va_test_ds[2])

# initialize dataloader
va_dataloader = init_dataloader(va_train_ds_al, BATCH_SIZE, type='random')

# run training
rmse = train(va_model, va_dataloader, EPOCHS, LR, device, joint=True)

