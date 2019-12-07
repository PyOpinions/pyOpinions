from random import Random

import numpy as np


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1)
    # if row_sums contains zeros, it would lead to NaN. If this occurs, check for 0 in the sums and raise exception
    matrix /= row_sums[:, np.newaxis]
    return matrix


def randomize_matrix(matrix: np.ndarray, random: Random):
    # TODO I may replace python random.Random with numpy random.RandomState later
    random.random()  # ensure different state every function call, even if the outer random is not called in between
    r = np.random.RandomState(random.getstate()[1])
    matrix[:, :] = r.random_sample(matrix.shape)[:, :]


def distance_between(arr1: np.ndarray, arr2: np.ndarray, f=np.linalg.norm):
    # this implementation is faster than using scipy.spatial.distance.euclidean(a, b)
    return f(arr1 - arr2)


def max_distance_between(positions1: np.ndarray, positions2: np.ndarray):
    max_dist, current_dist = 0.0, 0.0
    for i, position2 in enumerate(positions2):
        current_dist = distance_between(positions1[i], position2)
        max_dist = current_dist if current_dist > max_dist else max_dist
    return max_dist
