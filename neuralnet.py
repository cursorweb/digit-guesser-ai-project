from torch import nn
from torch.nn.functional import log_softmax


# test loader = [ 100 x ( [ batch_size x [ 1x28x28 image ] ], [ batch_size x label ] ) ]


# 32 == batch size
# train = [ 32 x [ 1x28x28 ] ]
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # converts to 32, 28^2
        self.relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # 512 is a constant
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.ReLU(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.relu_stack(x)
        return log_softmax(logits, dim=-1)
