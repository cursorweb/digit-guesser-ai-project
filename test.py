import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neuralnet import NeuralNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mac
print(f"Loaded {DEVICE} device")

ROWS = 4
COLS = 4


test_data = datasets.MNIST(
    "./data/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

test_loader = DataLoader(test_data, shuffle=True, batch_size=6 * 5)

examples = enumerate(test_loader)
_, (imgs, labels) = next(examples)

model = NeuralNetwork().to(DEVICE)
model.load_state_dict(torch.load("model.pth", weights_only=True))

model.eval()

import matplotlib.pyplot as plt

fig = plt.figure()
with torch.no_grad():
    for i in range(ROWS * ROWS):
        plt.subplot(ROWS, ROWS, i + 1)
        plt.tight_layout()

        img = imgs[i]
        label = labels[i]

        plt.imshow(img.squeeze(), cmap="gray", interpolation="none")

        img = img.to(DEVICE)
        pred = model(img)

        certainty, guess = pred.exp().max(1)
        certainty, guess = certainty.item(), guess.item()
        # certainty = 0  # pred.exp().max(1)[0]
        # guess = pred.argmax(1)[0]
        actual = label

        plt.title(f"Guess: {guess} ({certainty * 100:.2f})\n Actual: {actual}")
        plt.xticks([])
        plt.yticks([])

plt.show()
