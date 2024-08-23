import numpy as np

from torchvision import transforms

import torch

from neuralnet import NeuralNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mac
print(f"Loaded {DEVICE} device")

model = NeuralNetwork().to(DEVICE)
model.load_state_dict(torch.load("model.pth", weights_only=True))

model.eval()

import matplotlib.pyplot as plt

img = plt.imread("number.png")


def rgb2gray(rgb):
    return np.round(1 - np.mean(rgb, -1), decimals=1)


img = rgb2gray(img)


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

img = transform(img)

with torch.no_grad():
    img = img.to(DEVICE)
    pred = model(img)

    certainty, guess = pred.exp().max(1)
    certainty, guess = certainty.item(), guess.item()
    print("guess:", guess, "certainty:", certainty)

    plt.imshow(img.cpu().squeeze(), cmap="gray")
    plt.title(f"Guess: {guess} ({certainty * 100:.2f})")
    plt.xticks([])
    plt.yticks([])

plt.show()
