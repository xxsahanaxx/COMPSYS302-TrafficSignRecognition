import torch
import torch.nn as nn


# Architecture for the customised ResNet model inspired by : https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/CNN_architectures/pytorch_resnet.py

# Creating a ResNet model from scratch on PyTorch for our customised database using the research paper

class baseBlock(nn.Module):
    # Identity_downsample is a parameter for a convolutional layer that will be changed 
    #       if we change the number of input channels or input size 
    def __init__(self, input_channels, output_channels, identity_downsample=None, stride=1):
        super(baseBlock, self).__init__()
        self.factor = 4
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(output_channels)

        # Stride here is what gets passed into the block as a parameter and kernel size changes to 3 as per research paper map
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3,stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.conv3 = nn.Conv2d(output_channels, output_channels*self.factor, kernel_size=1,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(output_channels*self.factor)
        self.relu = nn.ReLU()

        # Identity_downsample is a convolutional layer to maintain shape later on in the layers
        self.identity_downsample = identity_downsample
       

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # After the block is run, we add the identity parameter to x 

        # We change the identity_downsample layer if we need to change shape (happens only for the first block of the ResNetModel)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        # The transformed layer is then added to the existing layers and activated
        x = x + identity
        x = self.relu(x)
        return x

class ResNetModel(nn.Module):
    def __init__(self, baseBlock, layers, image_channels, num_classes):
        super(ResNetModel, self).__init__()
        # Initialising modules for the network
        self.input_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Sequential layers: [3,4,6,3]
        # For our code, according to the research paper, layer1 will have three iterations of baseBlock (layers[0]), 
        #       layer2 will have 4, layer3 will have 6 and layer4 will have 3 iterations of baseBlock
        self.layer1 = self.__make_layer(baseBlock, layers[0], output_channels=64, stride=1)
        self.layer2 = self.__make_layer(baseBlock, layers[1], output_channels=128, stride=2)
        self.layer3 = self.__make_layer(baseBlock, layers[2], output_channels=256, stride=2)
        self.layer4 = self.__make_layer(baseBlock, layers[3], output_channels=512, stride=2)

        # Depending on the input provided, it changes the output to a particular size
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        # Initialising the forward function
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Send into ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Average pool the layers 
        x = self.avgpool(x)

        # Reshaping to fit into a fully connected layer
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    # num_residual blocks = no. of times to use the blocks (every element of the layers array)
    def __make_layer(self, baseBlock, num_residual_blocks, output_channels, stride):
        identity_downsample = None
        layers = []

        # We need to make sure the dimensions of identity_downsample's convolutional layer
        #       is consistent with the rest of our layers so that they can be added
        # Therefore, if stride changes or number of input channels is not the same as factor times output, 
        #       we need to recreate the convolutional layer
        if stride != 1 or self.input_channels != 4 * output_channels:
            identity_downsample = nn.Sequential(nn.Conv2d(self.input_channels, output_channels*4, kernel_size=1,stride=stride), 
                                                nn.BatchNorm2d(output_channels*4))

        # Appending the first residual block to change channels (making current output the next input)
        layers.append(baseBlock(self.input_channels, output_channels, identity_downsample, stride))

        # To change the number of channels, we use the following statement:
        self.input_channels = output_channels * 4

        # Since we already computed the first residual block to change channels, we reduce the range by 1
        for i in range(num_residual_blocks - 1): 
            layers.append(baseBlock(self.input_channels, output_channels))

        # Unpacking the layers list and making them sequential layers, to output one after the other
        return nn.Sequential(*layers)

def customResNetModel(img_channels=3, layers=[3,4,6,3], num_classes=43):
    return ResNetModel(baseBlock, layers, img_channels, num_classes)
