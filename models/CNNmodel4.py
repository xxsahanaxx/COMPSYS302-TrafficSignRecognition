import torch.nn as nn
import torch.nn.functional as F
import torch


# Model inspiration is from https://data-flair.training/blogs/python-project-traffic-signs-recognition/.

# The network should inherit from the nn.Module
class customNetwork4(nn.Module):
    def __init__(self):
        super(customNetwork4, self).__init__()
        
        # Defining two convolutional layers
        # Input: 3 channels => output 64 => 128 channels
        self.conv1 = nn.Conv2d(3, 64, 3, 1)  
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        
        # Maxpooling after the two layers of convolution ensures some downsampling is done, taking the kernel size as input
        self.maxpool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.maxpool3 = nn.MaxPool2d(2)
        
        # ReLU is our activation function
        self.relu = nn.ReLU()
        
        # Dropout layers are used to regularise the model and work around overfitting models. 
        # It will 'filter' out some of the outputs from the layer above it
        # 0.25 implies one fourth of the outputs are filtered while 0.5 means half the outputs are
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.5)
        self.dropout4 = nn.Dropout2d(0.25)

        # In order to save ourselves from the burden of calculations, we use AdaptiveAvgPool2d
        # that asks the output size from the user and according changes the kernel size and stride by itself 
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Output from average pool goes into fully connected layers
        # Fully connected layer: input size, output size
        self.fc1 = nn.Linear(6*6*256, 256)
        
        # fc1 must be activated and dropped out before being connected to fc2
        self.fc2 = nn.Linear(256, 43)

    # forward() links all layers together
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        x = self.avgpool(x)
        # before fully connected layer, the average pooled layer must be flattened to make it linear
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x
