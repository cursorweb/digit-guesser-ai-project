import torch

from torch.utils.data import DataLoader

from neuralnet import NeuralNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mac
print(f"Loaded {DEVICE} device")


test_data = torch.load("./test_set.pt", weights_only=False)

img, label = test_data.dataset[0]

model = NeuralNetwork().to(DEVICE)
model.load_state_dict(torch.load("model.pth", weights_only=True))

model.eval()

import matplotlib.pyplot as plt

fig = plt.figure()
with torch.no_grad():
    # img = imgs[0]
    # label = labels[0]

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
