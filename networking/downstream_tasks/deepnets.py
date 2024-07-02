"""
This script will be exclusively so we can get some sort of gradient feedback from the whole samplinge experience
"""

from torch import nn


class Classifier(nn.Module):

    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.net(x)
