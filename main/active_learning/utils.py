from sklearn.model_selection import train_test_split
import torch
import math
import numpy as np
import random
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn import MSELoss
from .algorithms import monte_carlo_dropout, farthest_first
import matplotlib.pyplot as plt
import pandas as pd

def seed_pool_split(input_ids, attention_masks, labels, seed_size, random_state):
  input_ids_seed, input_ids_pool = train_test_split(input_ids.numpy(), train_size=seed_size, random_state=random_state)
  attention_masks_seed, attention_masks_pool = train_test_split(attention_masks.numpy(), train_size=seed_size, random_state=random_state)
  labels_seed, labels_pool = train_test_split(labels.numpy(), train_size=seed_size, random_state=random_state)
  return (torch.LongTensor(input_ids_seed), torch.LongTensor(attention_masks_seed), torch.FloatTensor(labels_seed)), (torch.LongTensor(input_ids_pool), torch.LongTensor(attention_masks_pool), torch.FloatTensor(labels_pool))


def extract_embeddings(model, dataloader, device):
    # iterates through the dataloader to extract the embeddings from the last model layer
    embeddings = []

    with torch.no_grad():
        for i, (X, _, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            out = model(X)
            hidden_states = out['hidden_states']

            layer_idx = -1  # we want to extract the last layer
            # instance_idx = 0    # there is only 1 instance because batch_size = 1

            # loop through all the instances in each batch!
            for instance_idx in range(len(X)):
                # extract the sentence embedding
                sentence_embedding = hidden_states[layer_idx][instance_idx][0][:].to('cpu').numpy()
                # append to embeddings list
                embeddings.append(sentence_embedding)

    print("Number of embedding vectors: ", len(embeddings))
    embeddings_arr = np.array(embeddings)
    return embeddings_arr



def convert2float(tensor):
  if torch.cuda.is_available():
    tensor = tensor.type(torch.cuda.FloatTensor)
  else:
    tensor = tensor.type(torch.FloatTensor)
  return tensor



def init_dataloader(data, batch_size, type='random'):
    # function to keep seed deterministic (for reproducability)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)

    if type == 'random':
        dataloader = DataLoader(data, sampler=RandomSampler(data), batch_size=batch_size,
                                worker_init_fn=seed_worker, generator=g)

    if type == 'sequential':
        dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=batch_size,
                                worker_init_fn=seed_worker, generator=g)

    return dataloader


def train(model, dataloader, epochs, lr, device):
    size = len(dataloader.dataset)
    print(f"Dataset Size: {size} instances")

    cum_rmse = 0  # cumulative RMSE

    # initiate optimizer and loss, set model to training mode
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()
    model.train()                       # sets model into training mode

    for e in range(epochs):
        for X, _, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = convert2float(y)  # convert to float tensor
            out = model(X)  # forward pass
            pred = convert2float(out.logits.flatten())  # extract predictions, convert to Float

            loss = loss_fn(pred, y)  # compute loss
            optimizer.zero_grad()  # clean out previous gradients
            loss.backward()  # backpropagate loss
            optimizer.step()  # update weights

            # cumulate RMSE
            rmse = math.sqrt(loss.item())
            cum_rmse += rmse

    avg_rmse = (cum_rmse / size) / epochs
    print('Average RMSE: ', avg_rmse)
    return avg_rmse


def test(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    cum_rmse = 0 # cumulate test loss
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for X, _, y in dataloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            pred = out.logits.flatten()
            cum_rmse += math.sqrt(loss_fn(pred, y).item())
    avg_rmse = cum_rmse /size
    print(f"Average test RMSE: {avg_rmse}")
    return avg_rmse


# function to handle making choices and storing them
def make_choices(al_rank_idxs, batch_size, pool):
    """
    al_rank_idxs: an np.array of indices chosen by an active learning algorithm
    batch_size: int that represents size of one training batch (e.g. 16)
    pool: ALDataset object that represents the learning pool
    """
    chosen_idxs = al_rank_idxs[[i for i in range(batch_size)]]
    chosen_instances = pool[chosen_idxs]

    return chosen_idxs, chosen_instances


def experiment_AL(seed, pool, test_set, model, query_method, lr, batch_size, device):
    """
    seed: ALDataset object that contains the initial seed (size equals to the batch_size)
    pool: remaining part of the ALDataset
    test_set: TensorDataset that contains the testing data, which is not split into seed and pool
    model: the intialized BERT instance
    query_method: str describing the method, e.g., random vs. eff etc..
    lr: float representing the learning rate
    batch_size: int representing the batch size
    device: cpu or gpu
    """

    train_rmse_curve = []
    test_rmse_curve = []
    pool_size = len(pool)

    # number of iterations: how many batches we have left in the pool
    num_iters = int(pool_size / batch_size)
    print("Total number of batches in the pool: ", num_iters)

    for i in range(num_iters):
        print(f" Finding batch {i + 1}/{num_iters}...")

        # update pool size
        pool_size = len(pool)
        print("Current pool size: ", pool_size)

        # store active learning instances and indexes that are chosen
        #     chosen_instances = []
        #     chosen_indices = []

        # initialize dataloaders for seed and pool (needed for FF, MCD), NOT FOR SEED!
        # toDO: check if SequentialSampler is appropriate, or RandomSampler
        # seed_dl = torch.utils.data.DataLoader(seed, sampler=SequentialSampler(seed), batch_size=train_batch_size)
        # pool_dl = torch.utils.data.DataLoader(pool, sampler=SequentialSampler(pool), batch_size=train_batch_size)
        seed_dl = init_dataloader(seed, batch_size, type='random')
        pool_dl = init_dataloader(pool, batch_size, type='random')

        # Step 1: train model on seed
        epochs = 1
        rmse = train(model, seed_dl, epochs, lr, device)
        train_rmse_curve.append(rmse)

        # Step 1.1: test model
        # for testing: sample sequentially and 1-by-1
        #test_dl = torch.utils.data.DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=1)
        test_dl = init_dataloader(test_set, 1, 'sequential')
        test_loss_fn = MSELoss()
        test_rmse = test(model, test_dl, test_loss_fn, device)
        test_rmse_curve.append(test_rmse)

        # ALGO 1: RANDOM
        if query_method == "random":
            al_rank_idxs = np.array(random.sample(range(0, pool_size), batch_size))  # sample random indices

        # ALGO 2: FARTHEST FIRST
        elif query_method == "farthest-first":

            # extract the embeddings
            seed_embeddings = extract_embeddings(model, seed_dl, device)
            pool_embeddings = extract_embeddings(model, pool_dl, device)

            al_rank_idxs = farthest_first(seed_embeddings, pool_embeddings)  # returns ranks of indices (descending)

        # ALGO 3: MC DROPOUT
        elif query_method == "mc-dropout":
            pool_dl = init_dataloader(pool, 1,
                                      type='sequential')  # for mc-dropout, we pass instance 1 by 1 sequentially
            num_simulations = 15  # number of simulations
            al_rank_idxs = monte_carlo_dropout(pool_dl, num_simulations, model, device)

        chosen_idxs, chosen_instances = make_choices(al_rank_idxs, batch_size, pool)
        print(f"Method: {query_method}. Chosen indices: {chosen_idxs}")
        # Step 3: append instance to seed and delete from pool
        seed.append_instances(chosen_instances)
        pool.delete_instances(chosen_idxs)

    return train_rmse_curve, test_rmse_curve



def plot_AL_results(v_results, a_results, lr, batch_size):
    fig = plt.figure()
    ax1 = fig.add_axes([0, 1, 1, 1])
    ax2 = fig.add_axes([0, 0, 1, 1])
    ax3 = fig.add_axes([1, 1, 1, 1])
    ax4 = fig.add_axes([1, 0, 1, 1])

    # plot the training loss curve (valence)
    for method in v_results.keys():
        ax1.plot(v_results[method][batch_size][f'train_{lr}'], label=method)
    ax1.legend()
    ax1.set_title(f'Valence Train RMSE')

    # plot the testing loss curve (valence)
    for method in v_results.keys():
        ax2.plot(v_results[method][batch_size][f'test_{lr}'], label=method)
    ax2.legend()
    ax2.set_title(f'Valence Test RMSE')

    # plot the training loss curve (arousal)
    for method in a_results.keys():
        ax3.plot(a_results[method][batch_size][f'train_{lr}'], label=method)
    ax3.legend()
    ax3.set_title(f'Arousal Train RMSE')

    # plot the testing loss curve (valence)
    for method in a_results.keys():
        ax4.plot(a_results[method][batch_size][f'test_{lr}'], label=method)
    ax4.legend()
    ax4.set_title(f'Arousal Test RMSE')

    plt.savefig(f'va_al_experiment_lr_{lr}_bs_{batch_size}.jpg')
    #plt.show()


# make table with the average results

def make_results_table(results, methods, batch_sizes, lrs):
    results_table = {}   # dictionary to populate
    for method in methods:
      for batch_size in batch_sizes:
        for lr in lrs:
          results_table[f'{method}_{batch_size}_{lr}_train'] = np.mean(results[method][batch_size][f'train_{lr}'])
          results_table[f'{method}_{batch_size}_{lr}_test'] = np.mean(results[method][batch_size][f'test_{lr}'])
    results_df = pd.Series(results_table).to_frame('Avg. RMSE')
    return results_table


def tokenize_sentences(tokenizer, sentences, max_len):
    input_ids = []
    attention_masks = []
    # For every sentence...
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_len,  # Pad & truncate all sentences.
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

    return input_ids, attention_masks


def set_device():
    # put device onto GPU if available, else on CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    if torch.cuda.is_available():
      print(f"GPU name: {torch.cuda.get_device_name()}")

    return device
