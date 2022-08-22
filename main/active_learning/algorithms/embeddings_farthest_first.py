from scipy.spatial.distance import cosine
import numpy as np


# toDO: make this more efficient by initializing empty array first
def find_closest(unlabeled_instance, seed_embeddings):
    distances = []
    for labeled_instance in seed_embeddings:
      dist = cosine(labeled_instance, unlabeled_instance)
      distances.append(dist)
    return np.min(np.array(distances))


def farthest_first(seed_embeddings, pool_embeddings):
    closest_distances = []
    for unlabeled_instance in pool_embeddings:
        min_dist = find_closest(unlabeled_instance, seed_embeddings)
        closest_distances.append(min_dist)
    return np.argsort(np.array(closest_distances))[::-1]   # reverse argsort returns indices in decreasing order
