import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import matplotlib.pyplot as plt
import numpy as np


def rgb2gray(rgb):
    return 1 - np.mean(rgb, -1)


class CustomImageDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_json(annotations_file, orient="index")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx].name)
        image = plt.imread(img_path)
        image = rgb2gray(image)
        label = self.img_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
