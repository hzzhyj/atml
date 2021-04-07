import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import random
from torchvision import transforms, datasets

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

def load_celeba(path):
    # please download the dataset manually since google drive does not allow
    # for big file downloads this way
    image_size = (64, 64)
    celeba_data = ImageFolder(path, transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.ToTensor()
                                ]))
    return celeba_data

class AddUniformNoise(object):
    def __init__(self, low=-0.1, high=0.1):
        self.low = low
        self.high = high
        self.m = torch.distributions.uniform.Uniform(self.low, self.high)
        
    def __call__(self, tensor):
        return tensor + self.m.rsample(sample_shape=tensor.size())

class CustomDSpritesDataset(Dataset): 
    def __init__(self, dataset, length=None, transform=None):
        dataset.allow_pickle = True
        self.imgs = torch.from_numpy(dataset["imgs"])

        if length != None :
            self.length = length
        else : 
            self.length = self.imgs.size(0)

        indices = torch.randperm(self.imgs.size(0))[:self.length]

        self.data = self.imgs[indices]

        self.transform = transform
        self.nb_factors = len(dataset['metadata'][()][b'latents_sizes'])
        self.factors_sizes = dataset['metadata'][()][b'latents_sizes']
        self.factors_names = dataset['metadata'][()][b'latents_names']

    def __getitem__(self, i):
        img = self.data[i]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.length

    def latent_to_index(self, latents):
        factors_bases = np.concatenate((self.factors_sizes[::-1].cumprod()[::-1][1:],
                                    np.array([1,])))
        return np.dot(latents, factors_bases).astype(int)

    def sample_latent(self, sample_size, fixed_factor = None):
        samples = np.zeros((sample_size, self.nb_factors))
        for lat_i, lat_size in enumerate(self.factors_sizes):
            if fixed_factor == lat_i:
                fixed_value = np.random.randint(lat_size, size=1)
                samples[:, lat_i] = np.full(sample_size, fixed_value)
            else:
                samples[:, lat_i] = np.random.randint(lat_size, size=sample_size)
        return samples

    def simulate_images(self, sample_size, fixed_factor = None):
        latents_sampled = self.sample_latent(sample_size, fixed_factor = fixed_factor)
        indices_sampled = self.latent_to_index(latents_sampled)
        imgs_sampled = self.imgs[indices_sampled]
        return imgs_sampled

    def num_factors(self):
        return self.nb_factors
                           
    def factors_sizes_list(self):
        return self.factors_sizes                    

class CustomDSpritesDatasetFactorVAE(CustomDSpritesDataset):
    def __init__(self, dataset, length=None , transform=None):
        super(CustomDSpritesDatasetFactorVAE, self).__init__(dataset, length, transform)

    def __getitem__(self, i):
        j = random.randrange(len(self))
        img1 = self.data[i]
        img2 = self.data[j]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2
    
class CustomClassifierDataset(Dataset):
    def __init__(self, factors, z_diffs, transform=None):
        self.factors = factors
        self.z_diffs = z_diffs
        self.length = z_diffs.size(0)
        self.transform = transform
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        factor = self.factors[i]
        z_diff = self.z_diffs[i]
        if self.transform is not None:
            factor = self.transform(factor)
            z_diff = self.transform(z_diff)
        return z_diff, factor
