"""
Try various ways to load image and inspect pixel data
"""

import torch
from torch.utils.data.dataset import Subset

test_data: Subset = torch.load("./data/test_set.pt", weights_only=False)

x = test_data
img = x.dataset[0][0]

import numpy as np

np.savetxt("mnist-img.out", img.squeeze(), fmt="%.4f")

import matplotlib.pyplot as plt

# print(img[0])
# plt.imshow(img.squeeze())
# plt.show()

import matplotlib.image

matplotlib.image.imsave("name.png", img.squeeze(), cmap="gray")
