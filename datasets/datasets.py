import numpy as np
import torch
from torch.utils.data import DataLoader, random_split


""" 
divides the dataset into train and test
"""


def train_test_random_split(dataset, percentage: float, seed=None: int):
    split_idx = int(percentage * len(data))
    if seed is not None:
        return random_split(dataset, [split_idx, len(data)-split_idx],
                            generator=torch.Generator().manual_seed(seed))
    else:
        return random_split(dataset, [split_idx, len(data)-split_idx])


""" loads dsprites dataset
the downloaded dataset contains imgs, latents_values, latents_classes and metadata
if images_only is true then we only return imgs, otherwise the raw_dataset with
all mentioned fields will be returned
"""


def load_dsprites(images_only=True):
    url = "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    raw_dataset = np.load("./dsprites.npz", encoding="bytes")
    if images_only:
        return raw_dataset["imgs"]
    return raw_dataset
