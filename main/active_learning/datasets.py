# Active Learning Dataset class
"""
- stores input_ids(tokens), attention masks, labels in a Tensor Dataset
- implements len() function, and getitem to enable indexing
- implements function to append and delete instances to Tensor Dataset by indices

"""

import torch
from torch.utils.data import TensorDataset
import random


class ALDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.data = TensorDataset(self.input_ids, self.attention_masks, self.labels)

    def __repr__(self):
        return f"{(self.input_ids, self.attention_masks, self.labels)}"

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        idx: can be int (single index) or list (several indices)
        """
        return (self.input_ids[idx], self.attention_masks[idx], self.labels[idx])

    def append_instances(self, new_instances):
        """
        new_instances: a 3-tuple consisting of input_ids (tensor), attention_mask (tensor) and labels (tensor)
        """
        self.input_ids = torch.vstack((self.input_ids, new_instances[0]))
        self.attention_masks = torch.vstack((self.attention_masks, new_instances[1]))
        self.labels = torch.hstack((self.labels, new_instances[2]))

        self.data = ALDataset(self.input_ids, self.attention_masks, self.labels)
        return self.data

    def delete_instances(self, remove_idxs):
        """
        remove_idxs: a list of indices to be removed
        """

        mask = np.ones(len(self.input_ids), dtype=bool)
        mask[remove_idxs] = False

        self.input_ids = self.input_ids[mask]
        self.attention_masks = self.attention_masks[mask]
        self.labels = self.labels[mask]

        self.data = ALDataset(self.input_ids, self.attention_masks, self.labels)
        return self.data

    def subsample(self, sample_size, random_seed):
        """
        sample_size: int, the size of our sample
        random_seed: a random seed number for replicability
        """

        random.seed(random_seed)
        random_idxs = random.sample(range(0, len(self.input_ids)), sample_size)

        input_ids_sample = self.input_ids[random_idxs]
        attention_masks_sample = self.attention_masks[random_idxs]
        labels_sample = self.labels[random_idxs]

        sample = ALDataset(input_ids_sample, attention_masks_sample, labels_sample)
        return sample