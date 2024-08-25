VISUALIZE = True
ROWS = 5
COLS = 6

import numpy as np

from torchvision import transforms

import torch

from neuralnet import NeuralNetwork

import json


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mac
print(f"Loaded {DEVICE} device")

model = NeuralNetwork().to(DEVICE)
model.load_state_dict(torch.load("model.pth", weights_only=True))

model.eval()

import matplotlib.pyplot as plt


def rgb2gray(rgb):
    return 1 - np.mean(rgb, -1)


with open("./inputs/class.json") as f:
    classified = json.load(f)


acc_mnist = 0
acc_custom = 0

with torch.no_grad():
    for i in range(len(classified)):
        fname = f"{i + 1}.png"
        img = plt.imread(f"./inputs/data/{fname}")
        img = rgb2gray(img)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)]
        )

        img = transform(img)
        label = classified[fname]

        img = img.to(DEVICE)
        pred = model(img)

        _, guess = pred.exp().max(1)
        guess = guess.item()
        actual = label

        if str(guess) == actual:
            acc_mnist += 1

        # ----- part 2 -----

        img = plt.imread(f"./inputs/data/{fname}")
        img = rgb2gray(img)

        mean = np.mean(img)
        std = np.std(img)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        img = transform(img)
        label = classified[fname]

        img = img.to(DEVICE)
        pred = model(img)

        _, guess = pred.exp().max(1)
        guess = guess.item()

        if str(guess) == actual:
            acc_custom += 1

print(f"mnist accuracy: {acc_mnist / len(classified) * 100:.2f}")
print(f"cust. accuracy: {acc_custom / len(classified) * 100:.2f}")

if not VISUALIZE:
    exit()

plt.figure()
plt.suptitle("MNIST normalization")
with torch.no_grad():
    for i in range(ROWS * COLS):
        plt.subplot(ROWS, COLS, i + 1)
        plt.tight_layout()

        fname = f"{i + 1}.png"
        img = plt.imread(f"./inputs/data/{fname}")
        img = rgb2gray(img)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)]
        )

        img = transform(img)
        label = classified[fname]

        plt.imshow(img.squeeze(), cmap="gray", interpolation="none")

        img = img.to(DEVICE)
        pred = model(img)

        _, guess = pred.exp().max(1)
        guess = guess.item()
        actual = label

        plt.title(f"Guess: {guess}\n Actual: {actual}")
        plt.xticks([])
        plt.yticks([])

plt.figure()
plt.suptitle("Custom normalization")
with torch.no_grad():
    for i in range(ROWS * COLS):
        plt.subplot(ROWS, COLS, i + 1)
        plt.tight_layout()

        fname = f"{i + 1}.png"
        img = plt.imread(f"./inputs/data/{fname}")
        img = rgb2gray(img)

        mean = np.mean(img)
        std = np.std(img)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        img = transform(img)
        label = classified[fname]

        plt.imshow(img.squeeze(), cmap="gray", interpolation="none")

        img = img.to(DEVICE)
        pred = model(img)

        _, guess = pred.exp().max(1)
        guess = guess.item()
        actual = label

        plt.title(f"Guess: {guess}\n Actual: {actual}")
        plt.xticks([])
        plt.yticks([])

plt.show()
