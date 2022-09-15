

# Just some utils to reshape the results of training/validation/testing

import numpy as np
import torch

###############################################################################
def bringToNumpy(value):    
    return value.cpu().detach().numpy()

###############################################################################
# Function that takes a set of datapoints with three dimensions, such as
# (batch size x temporal length x state dimension)
# and flattens the first two together.
def flattenDataAlongFirstTwoDimensions_torch(data):
    
    data = torch.flatten(input = data, start_dim = 0, end_dim = 1)
    
    return data

# Select the indices that contain non-zero input in a data array structured 
# such as:
#  (flattened dimension x state dimension)
def selectNonZeroIndices2Ddata_torch(data):
    
    # Summing along the state dimension
    # this is to avoid a for loop in finding the non zero dimensions.
    # Dimensions that have all zeros along all the state dimension, will also 
    # have their sum being so.
    sumAlongStateDimension = torch.sum(data, dim = 1)
    # Finding indices of values that are non-zero
    nonZeroIndices         = torch.nonzero(sumAlongStateDimension)
    
    nonZeroIndices         = torch.squeeze(nonZeroIndices)

    return nonZeroIndices

def selectGivenIndicesData_torch(data, indices):
    
    selectedData = data[indices,:]
    
    return selectedData

###############################################################################
def reshape1DdataWithInverseIndexingAndNoTrajectoryDimensionAndOneValuePerCluster(data, inverseIndexing):
    
    data = np.asarray(np.squeeze(data))
    
    if data.ndim == 5:
    
        data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2],data.shape[3], data.shape[4]))
        data = data[inverseIndexing, :, :]
        data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
        
    elif data.ndim == 4:
        
        data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2],data.shape[3]))
        data = data[inverseIndexing, :]
    
    return data

def reshape1DdataWithInverseIndexingAndNoTrajectoryDimension(data, inverseIndexing):
    
    data = np.asarray(np.squeeze(data))
    data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2],data.shape[3]))
    data = data[inverseIndexing, :, :]
    data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]))
    
    return data

def reshape1DdataWithNoInverseIndexingAndNoTrajectoryDimension(data):
    
    data = np.asarray(np.squeeze(data))
    data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2],data.shape[3]))
    data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]))
    
    return data

def reshape1DdataWithInverseIndexingAndWithTrajectoryDimension(data, inverseIndexing, numberOfTrajectories):
    
    data = np.asarray(data)
    
    # Params
    #if data.ndim > 4:
    #    data = np.squeeze(data)
    
    # Separate the trajectories
    data = np.reshape(data,
                     (data.shape[0], 
                      numberOfTrajectories,
                      -1, 
                      data.shape[2],
                      data.shape[3]))
    
    # Bring axis of trajectories more inside
    data = np.swapaxes(data, 1,2)
    
    # Combine epoch axis and sequence axis
    data = np.reshape(data, 
                     (data.shape[0]*data.shape[1], 
                      data.shape[2],
                      data.shape[3], 
                      data.shape[4]))
    
    # Reinvert indexing
    data = data[inverseIndexing, :, :, :]
    
    # Now bring trajectory dimension outside
    data = np.swapaxes(data, 0, 1)
    
    # now bring together the sequences in a trajectory and the length of sequence
    data = np.reshape(data, 
                     (data.shape[0], 
                      data.shape[1]*data.shape[2], 
                      data.shape[3]))
    
    return data

def reshape1DdataWithNoInverseIndexingAndWithTrajectoryDimension(data, numberOfTrajectories):
    
    # Params
    data = np.asarray(np.squeeze(data))
    
    # Separate the trajectories
    data = np.reshape(data,
                     (data.shape[0], 
                      numberOfTrajectories,
                      -1, 
                      data.shape[2],
                      data.shape[3]))
    
    # Bring axis of trajectories more inside
    data = np.swapaxes(data, 1,2)
    
    # Combine epoch axis and sequence axis
    data = np.reshape(data, 
                     (data.shape[0]*data.shape[1], 
                      data.shape[2],
                      data.shape[3], 
                      data.shape[4]))
    
    # Now bring trajectory dimension outside
    data = np.swapaxes(data, 0, 1)
    
    # now bring together the sequences in a trajectory and the length of sequence
    data = np.reshape(data, 
                     (data.shape[0], 
                      data.shape[1]*data.shape[2], 
                      data.shape[3]))
    
    return data
