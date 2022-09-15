
import torch
import numpy as np
import os

from ConfigurationFiles          import Config_GPU           as ConfigGPU
from KVAE_models_and_utils_codes import ConfigurationHolder  as ConfigHolder
from KVAE_models_and_utils_codes import VAE 
from KVAE_models_and_utils_codes import ImagesHolder         as IH
from KVAE_models_and_utils_codes import Python_loading_utils as PLU

# This is the main file for the training of the KVAE
###############################################################################
# DEFINE PATH TO CONFIGURATION FOLDER
# Path to configuration file starting from repository base folder
VAEconfigurationPath  = 'ConfigurationFiles/Config_VAE.json'
###############################################################################
# Extracting the base folder path
pathWhereDatasetFolderIsDefined = 'ConfigurationFiles/BaseDataFolder.txt'
baseFolderPath = PLU.ExtractBaseFolderPath(pathWhereDatasetFolderIsDefined)
###############################################################################
# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)
###############################################################################
# Read the configuration dictionaries
configHolder = ConfigHolder.ConfigurationHolder.PrepareConfigHolderWithOutputFolderToAddBase(VAEconfigurationPath, 
                                                                                              baseFolderPath)
# Define the other paths
configHolder = PLU.RedefinePathsVAEAddingBaseFolderPath(configHolder, baseFolderPath)
config = configHolder.config
###############################################################################
# LOAD DATA
print('Loading Training data and shuffle...')
dataTrainingShuffled   = IH.ImagesHolder(config['training_data_file']  , dimension_x = config['dimension_x'] , 
                                         dimension_y = config['dimension_y'], image_channels = config['image_channels'])
dataTrainingShuffled.ShuffleData()
print('Loading Validation data ...')
dataValidation = IH.ImagesHolder(config['validation_data_file'], dimension_x = config['dimension_x'] , 
                                 dimension_y = config['dimension_y'], image_channels = config['image_channels'])
# Bring to torch
dataTrainingShuffled.BringDataToTorch()
dataValidation.BringDataToTorch()
###############################################################################
# TRAINING THE VAE
# First the VAE is created ...
num_filters = [int(f) for f in config['dim_filters'].split(',')]
# Create VAE
vae = VAE.VAE(z_dim          = config['dim_a'], 
              image_channels = config['image_channels'], 
              image_size     = [config['dimension_x'],config['dimension_y']], 
              dim_filters    = num_filters, 
              kernel         = config['filter_size'], 
              stride         = config['stride']).to(device)
vae.print()
vae.PrintVAELayers(config['image_channels']) 
# ... and then trained
summaryTrainingAllEpochs, summaryValidationAllEpochs, \
    summaryTrainingCurrentEpoch, summaryValidationCurrentEpoch = vae.PerformVAETrainingAndTesting(
        config, dataTrainingShuffled, dataValidation)
del dataTrainingShuffled
###############################################################################
# CHOOSE THE TRAINING EPOCH
# We choose the epoch of training at which the validation got the best results.
lossesOnValidation = summaryValidationAllEpochs.summary['Total_loss']
chosenEpoch        = np.argmin(lossesOnValidation)
# Load and save the VAE of that epoch
vae.load_state_dict(torch.load(config['output_folder'] + '/vae_' + str(chosenEpoch) + '.torch'))
torch.save(vae.state_dict(), config['output_folder'] + '/vae.torch')
# Save also in the folder where to put the final model
if not os.path.exists(config['output_folder_kvae']):
    os.makedirs(config['output_folder_kvae'])
torch.save(vae.state_dict(), config['output_folder_kvae'] + '/vae.torch')
vae.load_state_dict(torch.load(config['output_folder'] + '/vae.torch'))
vae.FreezeBatchNormLayers()
###############################################################################
# RETEST ON THE CHOSEN EPOCH
# Load data again (in order)
print('Loading Training data ...')
dataTraining = IH.ImagesHolder(config['training_data_file']  , dimension_x = config['dimension_x'] , 
                               dimension_y = config['dimension_y'], image_channels = config['image_channels'])
dataTraining.BringDataToTorch()
# We put the printing of losses at the end
if config['image_channels'] != 1:
    trainingImages = torch.swapaxes(dataTraining.images, 0, 1)
summaryTrainingCurrentEpoch   = vae.TestVAEOverEpoch(trainingImages, outputFolder = config['output_folder'], 
                                                   currentEpochNumber = config['only_vae_epochs'], 
                                                   batchSize = 1, alpha = config['alpha_VAEonly_training'])
if config['image_channels'] != 1:
    validationImages = torch.swapaxes(dataValidation.images, 0, 1)
summaryValidationCurrentEpoch = vae.TestVAEOverEpoch(validationImages, outputFolder = config['output_folder'], 
                                                   currentEpochNumber = config['only_vae_epochs'], 
                                                   batchSize = 1, alpha = config['alpha_VAEonly_training'])
###############################################################################
# SAVE both the result from training and validation on the chosen epoch in a 
# defined folder.
summaryTrainingCurrentEpoch.BringValuesToMatlab(outputFolder = config['output_folder'], dataStructure = 2, 
                                                batchSize = 1, filePrefix = 'TRAIN_')
summaryValidationCurrentEpoch.BringValuesToMatlab(outputFolder = config['output_folder'], dataStructure = 2, 
                                                batchSize = 1, filePrefix = 'TEST_')


