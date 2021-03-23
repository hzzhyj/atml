import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import random

""" 
divides the dataset into train and test
"""


def train_test_random_split(dataset, percentage: float, seed: int = None):
    split_idx = int(percentage * len(dataset))
    if seed is not None:
        return random_split(dataset, [split_idx, len(dataset)-split_idx],
                            generator=torch.Generator().manual_seed(seed))
    else:
        return random_split(dataset, [split_idx, len(dataset)-split_idx])


""" loads dsprites dataset
the downloaded dataset contains imgs, latents_values, latents_classes and metadata
if images_only is true then we only return imgs, otherwise the raw_dataset with
all mentioned fields will be returned
"""


def load_dsprites(path, images_only=True):
    url = "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    raw_dataset = np.load(path, encoding="bytes")
    if images_only:
        return raw_dataset["imgs"]
    return raw_dataset


class CustomDataset(Dataset):
    def __init__(self, imgs_, length , transform=None):
        self.length = length
        data = torch.from_numpy(imgs_).unsqueeze(1).float()
        indices = torch.randperm(data.size(0))[:length]
        self.data = data[indices]
        self.transform = transform

    def __getitem__(self, i):
        j = random.randrange(len(self))
        img1 = self.data[i]
        img2 = self.data[j]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2

    def __len__(self):
        return self.length