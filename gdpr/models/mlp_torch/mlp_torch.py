import torch
from torch import nn

class HiddenBlockReLU(nn.Module):
    def __init__(self, hidden_channels=256):
        super().__init__()

        self.linear = nn.Linear(hidden_channels, hidden_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=256, number_linear_blocks=2, class_anonymization_cardinal=4):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, hidden_channels)
        self.relu_1 = nn.ReLU()
        self.forward_part = nn.Sequential(*[HiddenBlockReLU(hidden_channels) for x in range(number_linear_blocks)])
        self.classifier = nn.Linear(hidden_channels, class_anonymization_cardinal)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.forward_part(x)
        x = self.classifier(x)

        return x
