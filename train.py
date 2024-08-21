import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neuralnet import NeuralNetwork

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 100
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mac
print(f"Training using {DEVICE} device")

train_data = datasets.MNIST(
    "./data/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

test_data = datasets.MNIST(
    "./data/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=TEST_BATCH_SIZE,
    shuffle=True,
)


model = NeuralNetwork().to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters())


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
    print(
        f"\nTest set:\n\tAccuracy: {100 * correct:>.2f}%, Avg. Loss: {test_loss:>8f}\n"
    )


for epoch in range(EPOCHS):
    print(f"{'-' * 10} Epoch: {epoch + 1} {'-' * 10}")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)

    torch.save(model.state_dict(), "./model.pth")
