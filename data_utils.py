import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from os import path

class FeatureDataset(Dataset):
    def __init__(self, pkl_file):
        """
        Args:
            csv_file (string): Path to the csv file with preprocessed image features.
        """

        self.data = pd.read_pickle(pkl_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        feature = torch.squeeze(torch.tensor(np.array(self.data.iloc[idx]['latents']), dtype=torch.float))
        one_hot = self.data.label.iloc[idx]
        return feature, torch.tensor(one_hot.astype(np.float32), dtype=torch.float)

class ImageDataset(Dataset):
    def __init__(self, csv_file, imdir =''):
        """
        Args:
            csv_file (string): Path to the csv file with image paths and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.CenterCrop(512),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
        self.dir = imdir
        self.data = pd.read_csv(csv_file)
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