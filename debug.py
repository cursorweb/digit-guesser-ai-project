import torch
import numpy as np

from neuralnet import NeuralNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mac
print(f"Loaded {DEVICE} device")

model = NeuralNetwork().to(DEVICE)
model.load_state_dict(torch.load("model.pth", weights_only=True))

model.eval()

import matplotlib.pyplot as plt

img = plt.imread("number.png")
print(img.shape)


def rgb2gray(rgb):
    return 1 - np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


img = rgb2gray(img)

img = np.expand_dims(img, axis=0)

img = torch.as_tensor(img).to(torch.float32)

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
