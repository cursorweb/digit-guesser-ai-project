import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


n_epochs = 3
batch_size_train = 32
batch_size_test = 100
learning_rate = 0.01
momentum = 0.5
log_interval = 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
        # return F.softmax(x)


test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "./data/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_test,
    shuffle=True,
)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

network_state_dict = torch.load("./models/model.pth", weights_only=True)
network.load_state_dict(network_state_dict)

import matplotlib.pyplot as plt

with torch.no_grad():
    output = network(example_data)

output = output.data.exp().max(1)
# output = output.data.max(1)

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap="gray", interpolation="none")

    certainty = output[0][i].item()
    number = output[1][i].item()

    real = example_targets[i].item()

    plt.title(
        f"Prediction: {number} ({certainty * 100:.2f}%)\n\
              Actual: {real}"
    )
    plt.xticks([])
    plt.yticks([])

plt.show()
