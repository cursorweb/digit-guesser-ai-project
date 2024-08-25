import torch
from torch import nn

from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms

from neuralnet import NeuralNetwork
from custom_dataset import CustomImageDataset


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)]
)


dataset1 = datasets.MNIST(root="./data", transform=transform, download=True)
dataset2 = CustomImageDataset(
    annotations_file="./inputs/class.json",
    img_dir="./inputs/data/",
    transform=transform,
)

dataset = ConcatDataset([dataset1, dataset2])

# import pandas as pd

# print(pd.read_json("./inputs/class.json", orient="index").iloc[21].name)
# print(pd.read_json("./inputs/class.json", orient="index").iloc[21, 0])
# exit()

import matplotlib.pyplot as plt

train_dataloader = DataLoader(dataset1, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
print(train_features.shape)
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
