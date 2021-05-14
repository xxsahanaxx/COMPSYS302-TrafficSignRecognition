import torch 
import torchvision
import torch.nn # importing neural network modules and its necessary functions
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd # for file reading 
import numpy as np
from PIL import Image # image manipulation
import random
import os
import matplotlib.pyplot as plt
import torch.utils.data
import torch.optim as optim  # for optimisation algorithms
from torch.optim.lr_scheduler import StepLR
from models.CNNModel1 import AlexNet
from models.CNNModel2 import customResNetModel
from models.CNNModel3 import CustomNet1
from models.CNNmodel4 import customNetwork4
from dataset_import import TrafficSignDataset
from sklearn import metrics

# Global Parameters
num_classes = 43
num_epochs = 15
learn_rate = 0.001
save_model = True
gamma = 0.7
training_loss = []
validation_loss = []
validation_accuracy = []
testing_accuracy = []


def train_cnn(model, device, train_loader, optimizer, epoch, flag):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        # data and target are batches of feature sets and target labels, and batch_idx is the current batch
        data, target = data.to(device), target.to(device)

        # We generate the output through our current model for the data batch we've obtained
        scores = model(data)

        # defining the loss function
        criterion = torch.nn.CrossEntropyLoss()

        # We obtain the loss for the current label, applying softmax and negative logarithmic function to it before 
        # appending to the losses for the current target
        loss = criterion(scores, target)
        losses.append(loss.item())

        # We zero the parameter gradients, so that during back-propagation the gradients don't add up
        optimizer.zero_grad()
        loss.backward()

        # To adjust the weights, we call optimizer.step()
        optimizer.step()
    print(f"Loss at epoch {epoch} is {sum(losses)/len(losses)}")
    if(flag == 0):
      training_loss.append(float(sum(losses))/float(len(losses)))
    else:
      validation_loss.append(float(sum(losses))/float(len(losses)))


def test(model, device, test_loader,flag):
    # Since we are in testing, the BatchNorm and Dropout layers will be deactivated. 
    # Hence, we call model.eval() to declare that and speed up the evaluations
    model.eval()

    num_correct = 0
    num_samples = 0
    
    # Setting torch gradient to no_grad, we preserve data from leaking out of test set and being trained
    # Disabling gradients ensures that computations occur faster and memory is minimally consumed
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            scores = model(data)
            _, predictions = torch.max(scores, 1)

            num_correct += (predictions == target).sum() 
            num_samples += predictions.size(0)

            # Append batch prediction results
            global list_preds, list_targets
            list_preds.extend(predictions.view(-1).cpu())
            list_targets.extend(target.view(-1).cpu())

        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples) * 100}")

    # Changing from evaluate mode to training mode ensures that no more testing occurs until the next function call
    model.train()

    # Switching between validation and testing accuracy results
    if flag == 0:
        validation_accuracy.append(float(num_correct) / float(num_samples) * 100)
    else:
        testing_accuracy.append(float(num_correct) / float(num_samples) * 100)


def main():
    #Boonlean variable to switch between models
    CNNModel1 = True
    CNNModel2 = False
    CNNModel3 = False
    CNNModel4 = False

    # Get the current working directory (cwd) : this becomes the root directory
    cwd = os.getcwd()
    print(cwd)

    # Set device to use GPU or CPU : since we cannot use CUDA, the device for our models = CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define transformations for our dataset according to the specifications of our models
    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Loading the data: since the dataset is a set of images, it needs to be converted into a tensor
    loadingDataset = TrafficSignDataset(csv_file='Train.csv', root_dir=cwd, transform=transformations)

    # Fixing the seed so the shuffle order will be the same everytime : necessary for validating our model
    indices = list(range(len(loadingDataset)))
    random.seed(42)
    random.shuffle(indices)

    # We split our training set into training and validation sets with:
    #   - 80% samples in the training set, and
    #   - 20 % in the validation set

    # Initialising the size of our training set
    train_size = int(0.8 * len(loadingDataset))  # has a length of 31367

    # Training dataset has the first 31367 elements of loadingDataset (first 80%)
    train_dataset_split = torch.utils.data.Subset(loadingDataset, indices[:train_size])

    # Validation dataset takes the last 20%
    validation_dataset_split = torch.utils.data.Subset(loadingDataset, indices[train_size:])

    # Loading the testDataset using the created class TrafficSignDataset while applying transformations to the images:
    testDataset = TrafficSignDataset(csv_file='Test.csv', root_dir=cwd, transform=transformations)

    # The training & validatiom datasets are already shuffled at this point and hence
    # shuffle parameter = False when loading into DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset_split, batch_size=64, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset_split, batch_size=32, shuffle=False)

    # Test has not been shuffled yet so can be shuffled
    test_loader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=32, shuffle=True)
    print("Loading complete")

    #################### Build network and run ###################################

    if CNNModel1:
        model = AlexNet()
    if CNNModel2:
        # In order to use an equivalent of ResNet34 for our project, we have layers = [3, 4, 6, 3] as per the research paper
        model = customResNetModel(img_channels=3, layers=[3, 4, 6, 3], num_classes=num_classes)
    if CNNModel3:
        model = CustomNet1()
    if CNNModel4:
        model = customNetwork4()
        
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # We iterate through every epoch and train the training set, the validation set and the test set
    for epoch in range(num_epochs):
        # when flag = 0, the training loss/validation accuracy should be appended in respective functions
        flag = 0 
        train_cnn(model, device, train_loader, optimizer, epoch, flag)
        test(model, device, validation_loader,flag)
        
        # when flag = 1, the validation loss/testing accuracy should be appended in respective functions
        flag = 1        
        train_cnn(model, device, validation_loader, optimizer, epoch,flag)

        test(model, device, test_loader,flag)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "./results/customresnetmodel34.pt")
    
    plt.plot(np.arange(0,15), training_loss, np.arange(0,15), validation_loss) 
    plt.title("Loss curve")
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.savefig(cwd+"/results/loss4curve.png", bbox_inches='tight')
    
    plt.plot(np.arange(0,15), testing_accuracy, np.arange(0,15), validation_accuracy) 
    plt.title("Test Accuracy Curve")
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy per epoch')
    plt.legend(['Testing', 'Validation'])
    plt.savefig(cwd+"/results/testvalaccuracy4curve.png", bbox_inches='tight')

    resultmat = metrics.confusion_matrix(list_targets, list_preds, labels=np.arange(0,num_classes))
    print(resultmat)
    
    print("All necessary files displayed!")
    print("Model executed!")


if __name__ == '__main__':
    main()
