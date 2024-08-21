import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neuralnet import NeuralNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mac
print(f"Loaded {DEVICE} device")


test_data = datasets.MNIST(
    "./data/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)


model = NeuralNetwork().to(DEVICE)
model.load_state_dict(torch.load("model.pth"))

model.eval()

import matplotlib.pyplot as plt

fig = plt.figure()
with torch.no_grad():
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()

        img, label = test_data[i]

        plt.imshow(img.squeeze(), cmap="gray", interpolation="none")

        img = img.to(DEVICE)
        pred = model(img)

        guess = pred.argmax(1)[0]
        actual = label

        plt.title(f"Guess: {guess}\n Actual: {actual}")
        plt.xticks([])
        plt.yticks([])

plt.show()
