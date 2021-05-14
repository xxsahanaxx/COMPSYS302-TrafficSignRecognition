import torch.nn as nn
import torch.nn.functional as F
import torch

# Model is inspired by https://data-flair.training/blogs/python-project-traffic-signs-recognition/.
# The network should inherit from the nn.Module

class CustomNet1(nn.Module):
    def __init__(self):
        super(CustomNet1, self).__init__()
        # Define 2D convolution layers
        # 3: input channels, 32: output channels, 5: kernel size, 1: stride
        # The size of input channel is 3 because all images are coloured
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        # It will 'filter' out some of the input by the probability(assign zero)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # Fully connected layer: input size, output size
        self.fc1 = nn.Linear(692224, 128)
        self.fc2 = nn.Linear(128, 43)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)  # activation function
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
