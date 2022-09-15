
# This is a simple script to load data from a set of folders into a torch array.
# This can be used to train a VAE.

###############################################################################

import torch
import numpy as np
import random

from torchvision import datasets
from torchvision import transforms

###############################################################################

class ImagesHolder(object):
    
    # dimension_x
    # dimension_y
    # grayscale_setting
    # numberOfImages
    # images
    
    ###########################################################################
    # Initialization
    
    # Transforms to apply on the images
    # INPUTS:
    # - dimension_x: to what dimension to reduce the image along x
    # - dimension_y: to what dimension to reduce the image along y
    # - image_channels: number of image channels (e.g., 1 -> grayscale, 3 -> RGB)
    def DefineTransforms(self, dimension_x, dimension_y, image_channels):
        
        if image_channels == 1:
            transformsToApply = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((dimension_x, dimension_y)),
                transforms.ToTensor(),
                ]) 
        else:   
            transformsToApply = transforms.Compose([
                transforms.Resize((dimension_x, dimension_y)),
                transforms.ToTensor(),
                ]) 
        
        return transformsToApply
    
    # Initialize the 'images' property with zero values.
    def InitializeImages(self):
        
        if self.image_channels == 1:
            self.images = np.zeros((self.numberOfImages, self.dimension_x, self.dimension_y))
        else:
            self.images = np.zeros((self.image_channels, self.numberOfImages, self.dimension_x, self.dimension_y))
        
        return
    
    # Fill the 'images' properties from a dataloader object.
    # INPUTS:
    # - dataloader
    def FillImagesFromDataloader(self, dataloader):
        
        # Number of images
        self.numberOfImages = len(dataloader)
        # Initializing a torch array to store the images
        self.InitializeImages()
        
        idx = 0
        for (images,_) in dataloader:
            if self.image_channels == 1:
                self.images[idx, :, :]    = np.asarray(images)[0, 0, :, :]
            else:
                self.images[:, idx, :, :] = np.asarray(images)[0, :, :, :]
            idx += 1

        return
    
    # Initialization.
    # INPUTS:
    # - dataFilePath: path to the folder where the images are located.
    #   Note: the images must be placed in SUBFOLDERS of this folder.
    # - dimension_x: to what dimension to reduce the image along x
    # - dimension_y: to what dimension to reduce the image along y
    # - image_channels: number of image channels (e.g., 1 -> grayscale, 3 -> RGB)
    def __init__(self, dataFilePath, dimension_x, dimension_y, image_channels):
        
        self.dimension_x = dimension_x
        self.dimension_y = dimension_y
        self.image_channels = image_channels
        
        # Defining the transforms to apply on the data (resizing, grayscaling, cropping etc...)
        transformsToApply = self.DefineTransforms(dimension_x, dimension_y, image_channels)
        
        # Extract dataset and dataloader
        dataset    = datasets.ImageFolder(root=dataFilePath, transform=transformsToApply)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        
        self.FillImagesFromDataloader(dataloader)
        
        return
    
    ###########################################################################
    # Shuffling
    
    # Shuffling the images and outputting the indices of shuffling used.
    # OUTPUTS:
    # - newIndices: indices of shuffling.
    def _CalculateNewShufflingIndices(self):
        
        indicesOrder = np.arange(self.numberOfImages)
        newIndices  = indicesOrder
        for i in range(len(indicesOrder)):
            pickValue = random.choice(indicesOrder)
            indexPickedValue = np.where(indicesOrder == pickValue)
            indicesOrder= np.delete(indicesOrder, (indexPickedValue[0]), axis=0)
            newIndices[i] = pickValue
        
        return newIndices
     
    # Shuffle the image data.
    # OUTPUTS:
    # - newIndices: indices of shuffling.
    def ShuffleData(self):
        
        # Calculate the indices for shuffling
        newIndices = self._CalculateNewShufflingIndices()
    
        # Change the data in the object
        if self.image_channels == 1:
            self.images    = self.images[newIndices,:,:]
        else:
            self.images    = self.images[:,newIndices,:,:]
        
        return newIndices
    
    ###########################################################################
    # Additional utils
    
    def BringDataToTorch(self):
        
        self.images = torch.from_numpy(self.images).float() # recast to float to avoid them becoming doubles
        
        return
    
    def print(self):
        
        print('Size of input images:')
        print(str(self.dimension_x) + 'x' + str(self.dimension_y))
        print('Grayscale?:')
        print(self.grayscale_setting)
        print('Number of images')
        print(self.numberOfImages)
        
        return
