import torch
from torch.utils.data.dataset import Subset

test_data: Subset = torch.load("./data/test_set.pt", weights_only=False)

x = test_data
img = x.dataset[0][0]

import matplotlib.pyplot as plt

print(img[0])
plt.imshow(img.squeeze())
plt.show()
