RETRAIN = True  # Should retrain existing model?


import torch
from torch import nn

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from neuralnet import NeuralNetwork

import os


TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 100
EPOCHS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mac
print(f"Training using {DEVICE} device")


try:
    raise Exception()
    train_set = torch.load("./data/train_set.pt", weights_only=False)
    val_set = torch.load("./data/val_set.pt", weights_only=False)
    print("Using ./data/train_set.pt")
except:
    if not os.path.isdir("./data"):
        os.makedirs("./data")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)]
    )
    dataset = datasets.MNIST(root="./data", transform=transform, download=True)
    train_set, val_set, test_set = random_split(dataset, [50000, 10000 - 50, 50])
    # torch.save(train_set, "./data/train_set.pt")
    # torch.save(val_set, "./data/val_set.pt")
    # torch.save(test_set, "./data/test_set.pt")
    print("Saved to ./data/train_set.pt")


train_loader = DataLoader(
    dataset=train_set,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    dataset=val_set,
    batch_size=TEST_BATCH_SIZE,
    shuffle=True,
)


model = NeuralNetwork().to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


if RETRAIN:
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    try:
        optimizer.load_state_dict(torch.load("optimizer.pth", weights_only=True))
    except:
        pass

loss_data = []
acc_data = []


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (img, label) in enumerate(dataloader):
        img, label = img.to(DEVICE), label.to(DEVICE)  # move from cpu ??? to gpu ???

        # get loss
        pred = model(img)
        loss = loss_fn(pred, label)

        # update model (backpropagation)
        loss.backward()  # calculate the gradient ??
        optimizer.step()  # backpropagate ??
        optimizer.zero_grad()  # reset gradient ??

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(img)
            loss_data.append(loss)
            print(
                f"\nloss: {loss:>7f}, [{current:>5d}/{size:>5d} ({current/size * 100:.2f}%)]",
                end="",
            )


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():  # no_grad when not training
        for img, label in dataloader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            pred = model(img)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= num_batches  # take average
    correct /= size
    acc_data.append(correct)
    print(
        f"\nTest set:\n\tAccuracy: {100 * correct:>.2f}%, Avg. Loss: {test_loss:>8f}\n"
    )


for epoch in range(EPOCHS):
    print(f"{'-' * 10} Epoch: {epoch + 1} {'-' * 10}")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)

    torch.save(model.state_dict(), "./model.pth")
    torch.save(optimizer.state_dict(), "./optimizer.pth")

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
plt.plot(range(len(loss_data)), loss_data)
plt.plot(np.arange(len(acc_data)) * (len(loss_data) // len(acc_data)), acc_data)
plt.show()
