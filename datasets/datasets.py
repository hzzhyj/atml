import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
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
        
    def __call__(self, tensor, latent_values=None):
        out = tensor + self.m.rsample(sample_shape=tensor.size())
        out = torch.clamp(out, min=0, max=1)
        return out

class AddGeneratedNoise(object):
    def __init__(self, path, device, epsilon=.1):
        self.epsilon = epsilon
        self.device = device
        self.noisenet = NoiseGeneratorNet()
        self.noisenet.load_state_dict(torch.load(path, map_location=self.device))
        self.noisenet = self.noisenet.to(self.device)
        self.noisenet.eval()
        
    def __call__(self, tensor, latent_values):
        with torch.no_grad():
            # noise = self.noisenet(latent_values[:, 1:].type(torch.float).to(self.device))
            # print(tensor.shape)
            # out = torch.clamp(tensor.to(self.device) + self.epsilon * noise, min=0, max=1).detach()
            out = tensor
            return out

class NoiseGeneratorNet(nn.Module):
    def __init__(self, max_norm=1):
        super(NoiseGeneratorNet, self).__init__()

        self.max_norm = max_norm
        self.net = nn.Sequential(
            nn.Linear(5, 1200),
            nn.ReLU(),
            nn.Linear(1200, 1200),
            nn.ReLU(),
            nn.Linear(1200, 64 * 64),
        )

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, 64, 64)
        out = torch.clamp(out, min=-self.max_norm, max=self.max_norm)

        return out

class CustomDSpritesDataset(Dataset): 
    def __init__(self, dataset, length=None , transform=None, transform_needs_latents=False, seed=None):
        dataset.allow_pickle = True
        self.seed = seed
        self.imgs = torch.from_numpy(dataset["imgs"]) # shape (n, 64, 64)

        if length != None :
            self.length = length
            if seed!=None:
                indices = torch.randperm(self.imgs.size(0), generator=torch.Generator().manual_seed(seed))[:self.length]
            else:
                indices = torch.randperm(self.imgs.size(0))[:self.length]
            self.data = self.imgs[indices]
            self.idx = indices
            
        else : 
            self.length = self.imgs.size(0)
            self.data = self.imgs
            self.idx = torch.arange(self.length)
        
        self.transform = transform
        self.transform_needs_latents = transform_needs_latents 

        self.nb_factors = len(dataset['metadata'][()][b'latents_sizes'])
        self.factors_sizes = dataset['metadata'][()][b'latents_sizes']
        self.factors_names = dataset['metadata'][()][b'latents_names']
        self.factors_bases = np.concatenate((self.factors_sizes[::-1].cumprod()[::-1][1:],
                                    np.array([1,])))

        self.posX = dataset['metadata'][()][b'latents_possible_values'][b'posX']
        self.posY = dataset['metadata'][()][b'latents_possible_values'][b'posY']
        self.scale = dataset['metadata'][()][b'latents_possible_values'][b'scale']
        self.shape = dataset['metadata'][()][b'latents_possible_values'][b'shape']
        self.orientation = dataset['metadata'][()][b'latents_possible_values'][b'orientation']

    
        # ('color', 'shape', 'scale', 'orientation', 'posX', 'posY')
        self.latent_values = np.zeros((40, 6)).astype(float) # 40 is the maximum variations in a latent factor
        self.latent_values[:len(self.shape), 1] = self.shape
        self.latent_values[:len(self.scale), 2] = self.scale
        self.latent_values[:len(self.orientation), 3] = self.orientation
        self.latent_values[:len(self.posX), 4] = self.posX
        self.latent_values[:len(self.posY), 5] = self.posY

        self.latent_matrix = self.indices_to_latent(self.idx).astype(np.uint8)
        columns = np.arange(self.nb_factors, dtype=np.intp)
        self.latent_columns = columns[np.newaxis, :]

    def __getitem__(self, i):
        img = self.data[i]
        if self.transform is not None:
            if self.transform_needs_latents:
                lt_mx = np.expand_dims(self.latent_matrix[i], axis=0)
                latent_values = self.retrieve_latent_values(lt_mx)
                img = self.transform(tensor=img, latent_values=latent_values)
            else:
                img = self.transform(img)
        return img

    def __len__(self):
        return self.length

    def latent_to_index(self, latents):
        return np.dot(latents, self.factors_bases).astype(int)

    def sample_latent(self, sample_size, fixed_factor=None):
        samples = np.zeros((sample_size, self.nb_factors))
        
        for lat_i, lat_size in enumerate(self.factors_sizes):
            if fixed_factor == lat_i:
                fixed_value = np.random.randint(lat_size, size=1)
                samples[:, lat_i] = np.full(sample_size, fixed_value)
            else:
                samples[:, lat_i] = np.random.randint(lat_size, size=sample_size)

        return samples

    def simulate_images(self, sample_size, fixed_factor=None):
        latents_sampled = self.sample_latent(sample_size, fixed_factor = fixed_factor)
        indices_sampled = self.latent_to_index(latents_sampled)
        imgs_sampled = self.imgs[indices_sampled]
        return imgs_sampled

    def num_factors(self):
        return self.nb_factors
                           
    def factors_sizes_list(self):
        return self.factors_sizes

    def retrieve_latent_values(self, latent_matrix):
        latent_matrix = latent_matrix.astype(np.intp)
        latent_values = self.latent_values[latent_matrix, self.latent_columns]

        # latent_matrix = latent_matrix.astype(np.int)
        # latent_values = np.zeros_like(latent_matrix).astype(np.float)
        
        # # ('color', 'shape', 'scale', 'orientation', 'posX', 'posY')
        # latent_values[:, 1] = self.shape[latent_matrix[:, 1]]
        # latent_values[:, 2] = self.scale[latent_matrix[:, 2]]
        # latent_values[:, 3] = self.orientation[latent_matrix[:, 3]]
        # latent_values[:, 4] = self.posX[latent_matrix[:, 4]]
        # latent_values[:, 5] = self.posY[latent_matrix[:, 5]]
        
        return torch.from_numpy(latent_values)

    def indices_to_latent(self, indices):
        subtract_previous = np.zeros(len(indices))
        latents = np.zeros((len(indices), self.nb_factors))
        for i in range(self.nb_factors):
            latents[:, i] = (indices - subtract_previous) // self.factors_bases[i]
            subtract_previous +=  latents[:, i] * self.factors_bases[i]

        return latents

class CustomDSpritesDatasetFactorVAE(CustomDSpritesDataset):
    def __init__(self, dataset, length=None , transform=None, seed= None):
        super(CustomDSpritesDatasetFactorVAE, self).__init__(dataset, length, transform,seed)
        if seed!= None:
            self.shuffled_indices = torch.randperm(self.length, generator=torch.Generator().manual_seed(seed))
        else:
            self.shuffled_indices = torch.randperm(self.length)
            
    def __getitem__(self, i):
        img1 = self.data[i]
        img2 = self.data[self.shuffled_indices[i]]
        idx1 = self.idx[i].item()
        idx2 = self.idx[self.shuffled_indices[i]].item()
        if self.transform is not None:
            latent_values1 = torch.from_numpy(self.indices_to_latent(np.array([idx1])))
            latent_values2 = torch.from_numpy(self.indices_to_latent(np.array([idx2])))
            img1 = self.transform(tensor=img1, latent_values=latent_values1)
            img2 = self.transform(tensor=img2, latent_values=latent_values2)
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
