
# This script contains functions for Loading Data given a configuration 
# dictionary including where the data is located itself.

# The two DataHolder classes (DataHolder and DataHolderLongSequences) are 
# called inside this code.

# This code allows to load training and testing data, depending on 
# - how the data is structured, i.e., if it is composed by a single trajectory
#   or by a set of trajectories;
# - if we want to use the data for training the VAE or the KVAE.
#   VAE:  better if data is completely shuffled and structured as 
#        [total_data_length, dimension_x, dimension_y] (for images)
#   KVAE: data needs to be in sequences, so it cannot be completely shuffled.
#         Modifications on the sequences can however be performed for them to
#         be presented in a continuous way.

###############################################################################
# Loading the DataHolder and DataHolderLongSequences classes
from KVAE_models_and_utils_codes import DataHolder              as DH
from KVAE_models_and_utils_codes import DataHolderNoSequence    as DHNS

###############################################################################
# Functions for loading a single data file and elaborate it in different ways,
# based on whether we want to shuffle it or not, and based on how the data 
# itself is structured.

# Loads data from a file, brings all non-parameter specific dimensions
# together and shuffles everything.
# To use to load data for VAE training.
# INPUTS:
# - dataFile: the file from where to read the data.
# OUTPUTS:
# - shuffledData: the loaded and shuffled data. The data is composed in the 
#   following way:
#   [total_data_length, dimension_x, dimension_y] (for images)
# - newIndicesShuffled: indices of changed data order
def LoadShuffledData(dataFile, image_channels):
    
    # Shuffled data
    shuffledData                = DH.DataHolder(dataFile = dataFile)
    shuffledData.Print()
    shuffledData                = shuffledData.BringTogetherAllNonParameterSpecificDimensions(image_channels = image_channels)
    newIndicesShuffled          = shuffledData.ShuffleDataEverything(image_channels = image_channels)
    
    return shuffledData, newIndicesShuffled

def LoadDataNoSequences(dataFile, image_channels):
    
    unshuffledData              = DHNS.DataHolderNoSequence(dataFile = dataFile, image_channels = image_channels)
    unshuffledData.BringDataToTorch()
    unshuffledData.Print()
    shuffledData                = DHNS.DataHolderNoSequence(dataFile = dataFile, image_channels = image_channels)
    shuffledData.BringDataToTorch()
    newIndicesShuffled          = shuffledData.ShuffleData()
    
    return unshuffledData, shuffledData, newIndicesShuffled

# Loads data from a file containing a single trajectory, and changes the order
# of the sequences to allow sequential filtering of the batches.
# INPUTS:
# - dataFile: the file from where to read the data.
#   Data inside should be structured in the following way:
#   [number of sequences, sequence length, dimension_x, dimension_y] (for images)
# OUTPUTS:
# - data: the loaded and modified data. The data will be structured again as 
#   [number of sequences, sequence length, dimension_x, dimension_y] (for images)
# - newIndices: indices of changed sequence order
# - inverseIndexing: inverse indices of 'newIndices'
def LoadDataForKVAESingleTrajectory(dataFile, batchSize, image_channels):
    
    # Data
    data                        = DH.DataHolder(dataFile = dataFile) 
    data.Print()
    newIndices, inverseIndexing = data.ChangeDataOrderToAllowSequentialFiltering(batchSize, image_channels) 
    
    return data, newIndices, inverseIndexing

###############################################################################
# Functions to load training and validation data, for VAE and KVAE training.

# Load the training and testing data for VAE.
# INPUTS:
# - config: configuration dictionary, that should have the following fields:
#           > 'training_data_file': path to the file of training data
#           > 'validation_data_file' : path to the file of validation data
# OUTPUTS:
# Outputs from 'LoadShuffledData', on both training and validation data.
def LoadTrainingAndValidationDataForVAE(config):
    
    print('training_data_file')
    print(config['training_data_file'])
    
    print('Load training data for VAE')
    shuffledTrainingData, newIndicesShuffled       = LoadShuffledData(dataFile = config['training_data_file'], 
                                                                      image_channels = config['image_channels'])
    
    print('testing_data_file')
    print(config['testing_data_file'])
    
    print('Load validdation data for VAE')
    shuffledValidationData, newIndicesShuffledValidation = LoadShuffledData(dataFile = config['validation_data_file'], 
                                                                      image_channels = config['image_channels'])

    return shuffledTrainingData, newIndicesShuffled, shuffledValidationData, newIndicesShuffledValidation

# Load the training and testing data for VAE.
# INPUTS:
# - config: configuration dictionary, that should have the following fields:
#           > 'training_data_file': path to the file of training data
#           > 'testing_data_file' : path to the file of testing data
#           > 'batch_size'        : desired batch size
#           > 'dataStructure'     : an integer indicating whether there is a 
#                                   single trajectory in the data (0) or a 
#                                   set of trajectories (1).
# OUTPUTS:
# Outputs from either 'LoadDataForKVAESingleTrajectory' (if data contains a 
# single trajectory) or 'LoadDataForKVAEMultipleTrajectories' (if data contains
# multiple trajectories), on both training and testing data. In the latter case,
# to remember that also the batch size is given.
def LoadTrainingAndValidationDataForKVAE(config):
    
    print('Extracting training data for KVAE')
    trainingData, newIndices, inverseIndexing = \
       LoadDataForKVAESingleTrajectory(dataFile  = config['training_data_file'], batchSize = config['batch_size'],
                                       image_channels = config['image_channels'])
      
    print('Extracting validation data for KVAE')
    validationData, newIndicesValidation, inverseIndexingValidation = \
       LoadDataForKVAESingleTrajectory(dataFile  = config['validation_data_file'], batchSize = config['batch_size'],
                                       image_channels = config['image_channels'])
    
    return config, trainingData, newIndices, inverseIndexing, \
           validationData, newIndicesValidation, inverseIndexingValidation
