import torch.nn as nn
import torch


# Model is inspired by https://medium.com/@kushajreal/training-alexnet-with-tips-and-checks-on-how-to-train-cnns-practical-cnns-in-pytorch-1-61daa679c74a
# The network should inherit from the nn.Module

class AlexNet(nn.Module):
    def __init__(self, init_weights=True):
        super(AlexNet, self).__init__()
        self.conv_base = nn.Sequential(
            # 3: input channels, 32: output channels, 5: kernel size, 1: stride
            # The size of input channel is 3 because all images are coloured
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False),
            # Normalize the input layer by adjusting and scaling the activations.
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            # # Maxpooling after one layer of convolution ensures some downsampling is done, taking the kernel size as input
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc_base = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # Fully connected layer: input size, output size
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 43),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv_base(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.fc_base(x)
        return x

    # weight initialization is to prevent layer activation outputs from exploding or vanishing
    # This is not in the existing AlexNet model
    def _initialize_weights(self):
        for m in self.modules():
            # Initialize weights for convolutional layers
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Initialize weights for batch normalization
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # Initialize weights for fully connected layers
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
