import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
from os import path
import random
import re

class FeatureDataset(Dataset):
    def __init__(self, pkl_file, npy_dir = None):
        """
        Args:
            pkl_file (string): Path to the csv file with preprocessed image features.
        """

        self.data = pd.read_pickle(pkl_file)
        print(self.data.columns)
        if npy_dir:
            self.data["latents"] = load_all_latents(npy_dir)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        feature = torch.squeeze(torch.tensor(np.array(self.data.iloc[idx]['latents']), dtype=torch.float))
        one_hot = self.data.label.iloc[idx]
        return feature, torch.tensor(one_hot.astype(np.float32), dtype=torch.float)

class ImageDataset(Dataset):
    def __init__(self, csv_file,
    imdir ='', 
    transform = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.CenterCrop(512),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    ):
        """
        Args:
            csv_file (string): Path to the csv file with image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.dir = imdir
        self.data = pd.read_csv(csv_file)
        self.labels = self.data.columns[3:]
        self.data["label"] = list(self.data.iloc[:, 3:].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.data.path.iloc[idx]
        image = Image.open(path.join(self.dir, img_name)).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = self.data.label.iloc[idx]
        label = torch.tensor(label.astype(np.float32), dtype=torch.float)
        
        return image, label

class SmoteishDataset(Dataset):
    def __init__(self, X, tail_indices, y, maps, indices):
        self.X = X
        self.tail_indices = tail_indices
        self.y = y
        self.maps = maps
        self.indices = indices

    def __len__(self):
        return len(self.tail_indices)

    def __getitem__(self, idx):
        reference = idx % len(self.tail_indices)
        neighbour = random.choice(self.indices[reference])
        X1 = self.X[self.tail_indices[reference]]
        X2 = self.X[neighbour]
        X1_map = self.maps[reference]
        X2_map = self.maps[neighbour]
        target = np.maximum(self.y[neighbour], self.y[reference])
        return X1, X2, X1_map, X2_map, target, reference

def load_all_latents(npy_dir):
    # List all .npy files in the output directory
    latent_files = sorted([os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith('.npy')], key=extract_number)

    # List to store the latent vectors
    all_latents = []

    # Load each file and append to the list
    for latent_file in latent_files:
        print(latent_file)
        latents = np.load(latent_file)
        all_latents.append(latents)

    # Concatenate all latents into a single array
    all_latents = list(np.concatenate(all_latents, axis=0))
    return all_latents

def extract_number(file_name):
        # Extract the number at the end of the file name before the .npy extension
        match = re.search(r'_(\d+)\.npy$', file_name)
        return int(match.group(1)) if match else -1

class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch