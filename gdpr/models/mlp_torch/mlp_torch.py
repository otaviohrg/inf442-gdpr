import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_channels=512, hidden_channels=256, number_linear_blocks=2, class_anonymization_cardinal=4):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, hidden_channels)
        self.forward_part = nn.Sequential(*[nn.Linear(hidden_channels, hidden_channels) for x in range(number_linear_blocks)])
        self.classifier = nn.Linear(hidden_channels, class_anonymization_cardinal)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.forward_part(x)
        x = self.classifier(x)

        return x
