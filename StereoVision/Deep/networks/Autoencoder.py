import torch
import torchvision
from torch import nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        # Second 2D convolutional layer, taking in the 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.unpool = nn.MaxUnpool2d(2, 2)

        # First fully connected layer
        self.fc1 = nn.Linear(20480, 4096)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(4096, 4096)

        # Decoder
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 20480)

        self.t_conv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(32, 1, 2, stride=2)

    def forward(self, x):
        # encoding
        x = F.relu(self.conv1(x))
        x, indices1 = self.pool(x)
        x = F.relu(self.conv2(x))
        x, indices2 = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # decoding
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(x.size(0), 256, 8, 10)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = self.unpool(x, indices2)
        x = F.relu(self.t_conv3(x))
        x = self.unpool(x, indices1)
        x = self.t_conv4(x)

        return x