# This script contains functions for the DataHolder CLASS
# This CLASS allows to read and memorize information related to some input data.
# Input data is visual data from cameras + odometry/distance data

###############################################################################

import matplotlib.pyplot as plt
import torch
import numpy as np
import random

from KVAE_models_and_utils_codes import Loading_utils as l_utils

from ConfigurationFiles import Config_GPU as ConfigGPU

###############################################################################
# ---------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)

###############################################################################


class DataHolder(object):
    
    def __init__(self, dataFile, trajectory_to_select = None):
        
        data  = l_utils.load_dict(dataFile)
            
        # Extract the features related to clustering
        self.Extract_overall_data_features(data, trajectory_to_select)
        
        # Bring data from numpy to torch so to be passed in pytorch
        # neural networks
        #self.BringDataToTorch()
        
        return
    
    def StandardizeParams(self, X_mean, X_std):
        
        for i in range(self.params.shape[2]):
            self.params[:,:,i] = self.params[:,:,i] - X_mean[i]
            self.params[:,:,i] = self.params[:,:,i] / X_std[i]
            
        return
    
    # Function to unroll the data so that instead of having
    # (num sequences, sequence length, dimension)
    # we have:
    # (num sequences * sequence length, dimension)
    def UnrollData(self, image_channels):
        
        # NUMPY ARRAY CASE
        
        if type(self.images)   == np.ndarray:
            self.controls = np.reshape(self.controls, 
                                    (self.controls.shape[0]*self.controls.shape[1], self.controls.shape[2]))
            self.odometry = np.reshape(self.odometry, 
                                    (self.odometry.shape[0]*self.odometry.shape[1], self.odometry.shape[2]))
            self.params   = np.reshape(self.params, 
                                    (self.params.shape[0]*self.params.shape[1], self.params.shape[2]))
            if hasattr(self, 'acceleration') and hasattr(self, 'orientation'):
                self.acceleration   = np.reshape(self.acceleration, 
                                    (self.acceleration.shape[0]*self.acceleration.shape[1], self.acceleration.shape[2]))
                self.orientation   = np.reshape(self.orientation, 
                                    (self.orientation.shape[0]*self.orientation.shape[1], self.orientation.shape[2]))
        
        if type(self.images)   == np.ndarray and image_channels == 1: # If numpy
            self.images   = np.reshape(self.images, 
                                    (self.images.shape[0]*self.images.shape[1], self.images.shape[2], 
                                     self.images.shape[3]))   
        elif type(self.images)   == np.ndarray and image_channels == 3: # If numpy
            self.images   = np.reshape(self.images, 
                                    (self.images.shape[0],self.images.shape[1]*self.images.shape[2], 
                                     self.images.shape[3], self.images.shape[4]))
            
        # TORCH TENSOR CASE
        
        if type(self.images) == torch.Tensor:
            self.controls = torch.squeeze(self.controls, dim = 0)
            self.controls = torch.reshape(self.controls, 
                                    (self.controls.shape[0]*self.controls.shape[1], self.controls.shape[2]))
            self.odometry = torch.squeeze(self.odometry, dim = 0)
            self.odometry = torch.reshape(torch.squeeze(self.odometry), 
                                    (self.odometry.shape[0]*self.odometry.shape[1], self.odometry.shape[2]))
            self.params   = torch.squeeze(self.params, dim = 0)
            self.params   = torch.reshape(self.params, 
                                    (self.params.shape[0]*self.params.shape[1], self.params.shape[2]))
            if hasattr(self, 'acceleration') and hasattr(self, 'orientation'):
                self.acceleration   = torch.squeeze(self.acceleration, dim = 0)
                self.acceleration   = torch.reshape(self.acceleration, 
                                    (self.acceleration.shape[0]*self.acceleration.shape[1], self.acceleration.shape[2]))
                self.orientation   = torch.squeeze(self.orientation, dim = 0)
                self.orientation   = torch.reshape(self.orientation, 
                                    (self.orientation.shape[0]*self.orientation.shape[1], self.orientation.shape[2]))
        
        self.images = torch.squeeze(self.images, dim = 0)
        
        if type(self.images) == torch.Tensor and image_channels == 1: # If torch
            self.images   = torch.reshape(self.images, 
                                    (self.images.shape[0]*self.images.shape[1], self.images.shape[2], 
                                     self.images.shape[3]))
        elif type(self.images) == torch.Tensor and image_channels == 3: # If torch
            self.images   = torch.reshape(self.images, 
                                    (self.images.shape[0], self.images.shape[1]*self.images.shape[2], 
                                     self.images.shape[3], self.images.shape[4]))

        return
    
    def BringTogetherAllNonParameterSpecificDimensions(self, image_channels):
        
        # 3 DIMS for parameters
        
        if self.images.ndim == 4 and image_channels == 1:
            self.images   = np.reshape(self.images, 
                                      (self.images.shape[0]*self.images.shape[1],
                                       self.images.shape[2],self.images.shape[3]))
        elif self.images.ndim == 5 and image_channels == 3:
            self.images   = np.reshape(self.images, 
                                      (self.images.shape[0], self.images.shape[1]*self.images.shape[2],
                                       self.images.shape[3],self.images.shape[4]))
        if self.params.ndim == 3:

            self.controls = np.reshape(self.controls, 
                                      (self.controls.shape[0]*self.controls.shape[1],
                                       self.controls.shape[2]))
            self.odometry = np.reshape(self.odometry, 
                                      (self.odometry.shape[0]*self.odometry.shape[1],
                                       self.odometry.shape[2]))
            self.params   = np.reshape(self.params, 
                                      (self.params.shape[0]*self.params.shape[1],
                                       self.params.shape[2]))
            
        # 4 DIMS for parameters 
        
        if self.images.ndim == 5 and image_channels == 1:
            self.images   = np.reshape(self.images, 
                                      (self.images.shape[0]*self.images.shape[1]*self.images.shape[2],
                                       self.images.shape[3],self.images.shape[4]))
        elif self.images.ndim == 6 and image_channels == 3:
            self.images   = np.reshape(self.images, 
                                      (self.images.shape[0], self.images.shape[1]*self.images.shape[2]*self.images.shape[3],
                                       self.images.shape[4],self.images.shape[5]))
        if self.params.ndim == 4:
        
            self.controls = np.reshape(self.controls, 
                                      (self.controls.shape[0]*self.controls.shape[1]*self.controls.shape[2],
                                       self.controls.shape[3]))
            self.odometry = np.reshape(self.odometry, 
                                      (self.odometry.shape[0]*self.odometry.shape[1]*self.odometry.shape[2],
                                       self.odometry.shape[3]))
            self.params   = np.reshape(self.params, 
                                      (self.params.shape[0]*self.params.shape[1]*self.params.shape[2],
                                       self.params.shape[3]))
        
        return self
    
    # Extracting the features from training and testing data
    # input:  KVAE (self)
    # output: KVAE (self) (with features extracted)
    def Extract_overall_data_features(self, data, image_channels, trajectory_to_select = None):
    
        # extract single dataset features
        self.d1              = data['d1'] # frame size x
        self.d2              = data['d2'] # frame size y
        # training data
        self.sequences = data['sequences'] # number of sequences
        self.images    = data['images']    # images
        self.controls  = data['controls']  # controls (=u)
        self.odometry  = data['odometry']  # odometry (could correspond to control)
        print('self odometry size {}'.format(self.odometry.shape))
        print('image channels {}'.format(image_channels))

        if 'acceleration' in data.keys():
            self.acceleration  = data['acceleration']  # linear acceleration 
        if 'orientation' in data.keys():
            self.orientation  = data['orientation']  # angular velocity 
            print('self orientation size {}'.format(self.orientation.shape))
        
        if data['images'].ndim  == 5 and image_channels == 1 and trajectory_to_select != None: # If this was saved as DataHolderLongSequences, just pick the first trajectory
            print('Dimension = 5, taking trajectory number ' + str(trajectory_to_select))

            self.images   = self.images[trajectory_to_select,:,:,:,:]
            self.controls = self.controls[trajectory_to_select,:,:,:]
            self.odometry = self.odometry[trajectory_to_select,:,:,:]
            if 'acceleration' in data.keys():
                self.acceleration = self.acceleration[trajectory_to_select,:,:,:]
            if 'orientation' in data.keys():
                self.orientation = self.orientation[trajectory_to_select,:,:,:]
        elif data['images'].ndim  == 5 and image_channels == 1 and trajectory_to_select == None:  
            print('Dimension = 5, taking all trajectories')
            
            self.images   = np.reshape(self.images, 
                                      (self.images.shape[0]*self.images.shape[1],
                                       self.images.shape[2],self.images.shape[3],self.images.shape[4]))
            self.controls = np.reshape(self.controls, 
                                      (self.controls.shape[0]*self.controls.shape[1],
                                       self.controls.shape[2],self.controls.shape[3]))
            self.odometry = np.reshape(self.odometry, 
                                      (self.odometry.shape[0]*self.odometry.shape[1],
                                       self.odometry.shape[2],self.odometry.shape[3]))
            
            if 'acceleration' in data.keys():
                self.acceleration = np.reshape(self.acceleration, 
                                      (self.acceleration.shape[0]*self.acceleration.shape[1],
                                       self.acceleration.shape[2],self.acceleration.shape[3]))
            if 'orientation' in data.keys():
                self.orientation = np.reshape(self.orientation, 
                                      (self.orientation.shape[0]*self.orientation.shape[1],
                                       self.orientation.shape[2],self.orientation.shape[3]))
            self.sequences = self.images.shape[0]
            
        if image_channels == 1:
            self.timesteps = self.images.shape[1] # number of time steps per sequence
        else:
            self.timesteps = self.images.shape[2] # number of time steps per sequence
        
        # If there was a key named 'params', insert it in params property, 
        # otherwise take the odometry data
        if 'params' in data.keys():
            self.params  = data['params']
            if data['images'].ndim  == 5 and image_channels == 1 and trajectory_to_select != None:
                self.params   = self.params[trajectory_to_select,:,:,:]
            if data['images'].ndim  == 5 and image_channels == 1 and  trajectory_to_select == None: 
                self.params = np.reshape(self.params, 
                                        (self.params.shape[0]*self.params.shape[1],
                                         self.params.shape[2],self.params.shape[3]))
        else:
            self.params  = data['odometry']
            if data['images'].ndim  == 5 and image_channels == 1 and  trajectory_to_select != None:
                self.params   = self.params[trajectory_to_select,:,:,:]
            if data['images'].ndim  == 5 and image_channels == 1 and  trajectory_to_select == None: 
                self.params = np.reshape(self.params, 
                                        (self.params.shape[0]*self.params.shape[1],
                                         self.params.shape[2],self.params.shape[3]))
                
        if 'startingPoints' in data.keys():
            self.startingPoints = data['startingPoints']
            
        # This tells us how many dimensions there are in the DataHolder
        self.numberOfDimensions = self.odometry.ndim

        return
    
    # function to keep only a certain number of sequences of the data
    def Cut_number_of_sequences(self, numberOfSequencesToKeep):
        
        self.sequences = numberOfSequencesToKeep # number of sequences
        self.images    = self.images[0:numberOfSequencesToKeep, :, :, :]    # images
        self.controls  = self.controls[0:numberOfSequencesToKeep, :, :]  # controls (=u)
        self.odometry  = self.odometry[0:numberOfSequencesToKeep, :, :]  # odometry (could correspond to control)
        
        return
    
    # This function modifies the sequencing of the data so that 
    # each sequence in the batch is not followed by the one that comes
    # directly after it, but by the one that comes a number of sequences
    # afterwards. This allows each output of one batch to succeed the output
    # of another one and the filtering to continue without interruptions
    # in the middle.
    # E.g., if the original batches were as follows:
    # 1 -> 2 -> 3 -> 4 ->
    # and batch size is = 2, 
    # instead of dividing them as
    # [1, 2] -> [2, 3]
    # we divide them as
    # [1, 3] -> [2, 4]
    # so that 2 follows 1 and 4 follows 3 in processing across batches.
    # NOTE: This is irrelevant for case with clustering version of code, 
    # but should be used with LSTM case!
    # Inputs:
    # - batchSize: batch size
    # Outputs:
    # - newIndices: indices of where the original data have been moves
    # - inverseIndexing: inverse indexing to go back to original indexing
    #   (for plotting and data saving)
    def ChangeDataOrderToAllowSequentialFiltering(self, batchSize, image_channels):
        
        numberOfSequences = self.sequences
        numberOfBatches   = numberOfSequences // batchSize # floor taken
        
        # Some sequences will be thrown away (the last ones) if they don't constitute
        # together one last batch
        divisionNumberFinal = batchSize*numberOfBatches
        
        newIndices      = np.zeros(divisionNumberFinal)
        inverseIndexing = np.zeros(divisionNumberFinal)
        
        indexDestination = 0
        for i in range(numberOfBatches):
            
            for j in range(batchSize):
                
                indexToInsert                = int(numberOfBatches*j + i)
                newIndices[indexDestination] = int(indexToInsert)
                
                # Also inverse indexing is need to go back to original order
                inverseIndexing[indexToInsert] = int(indexDestination)
                
                indexDestination += 1
                
        # this are the indices for going to the batch sequencing that we need
        newIndices         = newIndices.astype(int)
        
        # inverse indexing for going back to original sequencing (for plotting)
        inverseIndexing    = inverseIndexing.astype(int)
        
        self.images   = np.squeeze(self.images)
        self.controls = np.squeeze(self.controls)
        self.odometry = np.squeeze(self.odometry)
        self.params   = np.squeeze(self.params)
        # Change the data in the object
        if image_channels == 1:
            self.images    = self.images[newIndices, :,:,:]
        elif image_channels == 3:
            self.images    = self.images[:,newIndices, :,:,:]
        self.controls  = self.controls[newIndices,:]
        self.odometry  = self.odometry[newIndices,:]
        self.params    = self.params[newIndices,:]
        
        return newIndices, inverseIndexing
    
    def ShuffleData(self, image_channels):
        
        indicesOrder = np.arange(self.images.shape[0])
        
        newIndices  = indicesOrder
        for i in range(len(indicesOrder)):
            pickValue = random.choice(indicesOrder)
            indexPickedValue = np.where(indicesOrder == pickValue)
            indicesOrder= np.delete(indicesOrder, (indexPickedValue[0]), axis=0)
            newIndices[i] = pickValue
    
        # Change the data in the object
        if image_channels == 1:
            self.images    = self.images[newIndices, :,:,:]
        elif image_channels == 3:
            self.images    = self.images[:,newIndices, :,:,:]
        self.controls  = self.controls[newIndices,:]
        self.odometry  = self.odometry[newIndices,:]
        self.params    = self.params[newIndices,:]
        
        return newIndices
    
    # This shuffles everything (sequences and inside the sequences)
    def ShuffleDataEverything(self, image_channels):
        
        if self.images.ndim == 3 and image_channels == 1:
            indicesOrder            = np.arange(self.images.shape[0])
        elif self.images.ndim == 4 and image_channels == 1:
            indicesOrderSequences   = np.arange(self.images.shape[0])
            indicesOrderInSequences = np.arange(self.images.shape[1])
        elif self.images.ndim != 3 and self.images.ndim != 4 and image_channels == 1:
            indicesOrderSequences   = np.arange(self.images.shape[1])
            indicesOrderInSequences = np.arange(self.images.shape[2])
        elif self.images.ndim == 4 and image_channels == 3:
            indicesOrder            = np.arange(self.images.shape[1])
        elif self.images.ndim == 5 and image_channels == 3:
            indicesOrderSequences   = np.arange(self.images.shape[1])
            indicesOrderInSequences = np.arange(self.images.shape[2])
        elif self.images.ndim != 4 and self.images.ndim != 5 and image_channels == 3:
            indicesOrderSequences   = np.arange(self.images.shape[2])
            indicesOrderInSequences = np.arange(self.images.shape[3])
            
        #######################################################################
            
        if (self.images.ndim == 3 and image_channels == 1) or \
           (self.images.ndim == 4 and image_channels == 3):
            
            newIndices           = indicesOrder
            for i in range(len(indicesOrder)):
                pickValue = random.choice(indicesOrder)
                indexPickedValue = np.where(indicesOrder == pickValue)
                indicesOrder= np.delete(indicesOrder, (indexPickedValue[0]), axis=0)
                newIndices[i] = pickValue
            
        else:
            newIndicesSequences  = indicesOrderSequences
            for i in range(len(indicesOrderSequences)):
                pickValue = random.choice(indicesOrderSequences)
                indexPickedValue = np.where(indicesOrderSequences == pickValue)
                indicesOrderSequences= np.delete(indicesOrderSequences, (indexPickedValue[0]), axis=0)
                newIndicesSequences[i] = pickValue
                
            newIndicesOrderInSequences  = indicesOrderInSequences
            for i in range(len(indicesOrderInSequences)):
                pickValue = random.choice(indicesOrderInSequences)
                indexPickedValue = np.where(indicesOrderInSequences == pickValue)
                indicesOrderInSequences= np.delete(indicesOrderInSequences, (indexPickedValue[0]), axis=0)
                newIndicesOrderInSequences[i] = pickValue
                
        #######################################################################
    
        # Change the data in the object
        if self.images.ndim == 3 and image_channels == 1:
            self.images    = self.images  [newIndices,:,:]
            self.controls  = self.controls[newIndices,:]
            self.odometry  = self.odometry[newIndices,:]
            self.params    = self.params  [newIndices,:]
            
        elif self.images.ndim == 4 and image_channels == 1:
            self.images    = self.images  [newIndicesSequences,:,:,:]
            self.controls  = self.controls[newIndicesSequences,:,:]
            self.odometry  = self.odometry[newIndicesSequences,:,:]
            self.params    = self.params  [newIndicesSequences,:,:]
            
            self.images    = self.images  [:,newIndicesOrderInSequences,:,:]
            self.controls  = self.controls[:,newIndicesOrderInSequences,:]
            self.odometry  = self.odometry[:,newIndicesOrderInSequences,:]
            self.params    = self.params  [:,newIndicesOrderInSequences,:]
        elif self.images.ndim != 3 and self.images.ndim != 4 and image_channels == 1:
            self.images    = self.images  [:,newIndicesSequences,:,:,:]
            self.controls  = self.controls[:,newIndicesSequences,:,:]
            self.odometry  = self.odometry[:,newIndicesSequences,:,:]
            self.params    = self.params  [:,newIndicesSequences,:,:]
            
            self.images    = self.images  [:,:,newIndicesOrderInSequences,:,:]
            self.controls  = self.controls[:,:,newIndicesOrderInSequences,:]
            self.odometry  = self.odometry[:,:,newIndicesOrderInSequences,:]
            self.params    = self.params  [:,:,newIndicesOrderInSequences,:]
        elif self.images.ndim == 4 and image_channels == 3:
            self.images    = self.images  [:,newIndices,:,:]
            self.controls  = self.controls[newIndices,:]
            self.odometry  = self.odometry[newIndices,:]
            self.params    = self.params  [newIndices,:]
            
        elif self.images.ndim == 5 and image_channels == 3:
            self.images    = self.images  [:,newIndicesSequences,:,:,:]
            self.controls  = self.controls[newIndicesSequences,:,:]
            self.odometry  = self.odometry[newIndicesSequences,:,:]
            self.params    = self.params  [newIndicesSequences,:,:]
            
            self.images    = self.images  [:,:,newIndicesOrderInSequences,:,:]
            self.controls  = self.controls[:,newIndicesOrderInSequences,:]
            self.odometry  = self.odometry[:,newIndicesOrderInSequences,:]
            self.params    = self.params  [:,newIndicesOrderInSequences,:]
        elif self.images.ndim != 4 and self.images.ndim != 5 and image_channels == 3:
            self.images    = self.images  [:,:,newIndicesSequences,:,:,:]
            self.controls  = self.controls[:,newIndicesSequences,:,:]
            self.odometry  = self.odometry[:,newIndicesSequences,:,:]
            self.params    = self.params  [:,newIndicesSequences,:,:]
            
            self.images    = self.images  [:,:,:,newIndicesOrderInSequences,:,:]
            self.controls  = self.controls[:,:,newIndicesOrderInSequences,:]
            self.odometry  = self.odometry[:,:,newIndicesOrderInSequences,:]
            self.params    = self.params  [:,:,newIndicesOrderInSequences,:]
            
        #######################################################################
        
        if (self.images.ndim == 3 and image_channels == 1) or \
           (self.images.ndim == 4 and image_channels == 3):
            return newIndices
        else:
            return newIndicesSequences, newIndicesOrderInSequences
    
    def BringDataToTorch(self):
        
        self.images   = torch.from_numpy(self.images).float().to(device) # recast to float to avoid them becoming doubles
        self.controls = torch.from_numpy(self.controls).float().to(device)
        self.odometry = torch.from_numpy(self.odometry).float().to(device)
        self.params   = torch.from_numpy(self.params).float().to(device)
        if hasattr(self, 'acceleration') and hasattr(self, 'orientation'):
            self.acceleration = torch.from_numpy(self.acceleration).float().to(device)
            self.orientation = torch.from_numpy(self.orientation).float().to(device)
        
        return
    
    def Print(self):
        
        print('Size of input images:')
        print(str(self.d1) + 'x' + str(self.d2))
        
        print('Number of sequences:')
        print(self.sequences)
        print('Length of sequences:')
        print(self.timesteps)
        print('Odometry size:')
        print(self.odometry.shape[2])
        print('Parameters size')
        print(self.params.shape[2])
        
        return
    
    def PlotXYData(self):
        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3))
        
        for i in range(self.sequences):
            plt.scatter(self.odometry[i, :, 0], self.odometry[i, :, 1], color = 'red')
        
        return