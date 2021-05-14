import torch 
import torchvision
import torch.nn # importing neural network modules and its necessary functions
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
import torch.optim # for optimisation algorithms
import numpy as np
from PIL import Image
import random
import os
import matplotlib.pyplot as plt

# This class is to read csv files and assign labels with images
class TrafficSignDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # returns a 2D data structure with labelled axes
        self.annotations = pd.read_csv(csv_file)
        # sets root directory to the directory that is passed through
        self.root_dir = root_dir
        # if a transformation exists, then it is applied
        self.transform = transform

    def __len__(self):
        return len(self.annotations)  # should be 39209 images for our total set

    def __getitem__(self, index):
        # should return a specific image and target (?) to that image 
        # joins root_dir to annotations' row index and column 7 : name of the image (path) is there
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 7])
        curr_image=np.asarray(Image.open(img_path))
        # assigns an image to curr_image in the form 'RGB' channels
        curr_image = Image.fromarray(curr_image, 'RGB')
        curr_label = int(self.annotations.iloc[index, 6])
        # converting label to tensor
        curr_label = torch.tensor(curr_label)

        if self.transform is not None:
            curr_image = self.transform(curr_image)

        # return one image and its corresponding label
        return (curr_image, curr_label)


