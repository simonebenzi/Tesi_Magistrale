# This script contains functions for the DataHolder CLASS
# This CLASS allows to read and memorize information related to some input data.
# Input data is visual data from cameras + odometry/distance data

###############################################################################

import matplotlib.pyplot as plt
import torch
import numpy as np
import random

from KVAE_models_and_utils_codes import Loading_utils as l_utils

###############################################################################


class DataHolderNoSequence(object):
    
    def __init__(self, dataFile, image_channels, trajectory_to_select = None):
        
        data  = l_utils.load_dict(dataFile)
            
        # Extract the features related to clustering
        self.Extract_overall_data_features(data, trajectory_to_select)
        
        self.image_channels = image_channels
        
        return
    
    def StandardizeParams(self, X_mean, X_std):
        
        for i in range(self.params.shape[1]):
            self.params[:,i] = self.params[:,i] - X_mean[i]
            self.params[:,i] = self.params[:,i] / X_std[i]
            
        return
    
    def ExtractDatapointsAtGivenIndices(self, indices):
        
        if self.image_channels == 1:
            extractedImages     = self.images[indices, :, :]
        else:
            extractedImages     = self.images[:, indices, :, :]
            
        extractedControls       = self.controls[indices, :]
        extractedOdometry       = self.odometry[indices, :]
        extractedParams         = self.params[indices, :]
        
        return extractedImages, extractedControls, extractedOdometry, extractedParams
    
    # Extracting the features from training and testing data
    # input:  KVAE (self)
    # output: KVAE (self) (with features extracted)
    def Extract_overall_data_features(self, data, trajectory_to_select = None):
    
        # extract single dataset features
        self.d1              = data['d1'] # frame size x
        self.d2              = data['d2'] # frame size y
        # training data
        self.startingPoints = data['startingPoints'] # number of sequences
        self.images    = data['images']    # images
        self.controls  = data['controls']  # controls (=u)
        self.odometry  = data['odometry']  # odometry (could correspond to control)
        
        # If there was a key named 'params', insert it in params property, 
        # otherwise take the odometry data
        if 'params' in data.keys():
            self.params  = data['params']
        else:
            self.params  = data['odometry']
          
        # This tells us how many dimensions there are in the DataHolder
        self.numberOfDimensions = self.odometry.ndim
        
        return
    
    def ShuffleData(self):
        
        indicesOrder = np.arange(self.odometry.shape[0])
        
        newIndices  = indicesOrder
        for i in range(len(indicesOrder)):
            pickValue = random.choice(indicesOrder)
            indexPickedValue = np.where(indicesOrder == pickValue)
            indicesOrder= np.delete(indicesOrder, (indexPickedValue[0]), axis=0)
            newIndices[i] = pickValue
    
        # Change the data in the object
        if self.image_channels == 1:
            self.images    = self.images[newIndices,:,:]
        elif self.image_channels == 3:
            self.images    = self.images[:,newIndices,:,:]
        self.controls  = self.controls[newIndices,:]
        self.odometry  = self.odometry[newIndices,:]
        self.params    = self.params[newIndices,:]
        
        return newIndices
    
    def BringDataToTorch(self):
        
        self.images         = torch.from_numpy(self.images).float() # recast to float to avoid them becoming doubles
        self.controls       = torch.from_numpy(self.controls).float()
        self.odometry       = torch.from_numpy(self.odometry).float()
        self.params         = torch.from_numpy(self.params).float()
        self.startingPoints = torch.from_numpy(self.startingPoints).float()
        
        return
    
    def Print(self):
        
        print('Size of input images:')
        print(str(self.d1) + 'x' + str(self.d2))
        
        print('Odometry size:')
        print(self.odometry.shape[1])
        print('Parameters size')
        print(self.params.shape[1])
        
        return
    
    def PlotXYData(self):
        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3))
        
        for i in range(self.sequences):
            plt.scatter(self.odometry[i, :, 0], self.odometry[i, :, 1], color = 'red')
        
        return