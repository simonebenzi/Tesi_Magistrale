
# Load image and odometry data, combine them and put them together in a single
# trajectory, divided in batches of sequences.

###############################################################################
# IMPORTS
import os

from KVAE_models_and_utils_codes import ConfigurationHolder  as ConfigHolder
from KVAE_models_and_utils_codes import Python_loading_utils as PLU
from KVAE_models_and_utils_codes import Extract_data_utils   as EDU

###############################################################################
# This will print the indices of the loop 
check_mode = True
###############################################################################
# Folders definition
# Path to configuration file of KVAE
KVAEconfigurationPath  = 'ConfigurationFiles/Config_KVAE.json'
# Path to .txt file containing the path to the dataset folder
pathWhereDatasetFolderIsDefined = 'ConfigurationFiles/BaseDataFolder.txt'
baseFolderPath = PLU.ExtractBaseFolderPath(pathWhereDatasetFolderIsDefined)
# Path where to save the output of this code
path_to_inputs_to_final_model_folder = baseFolderPath + '/InputsToFinalModel/'
if not os.path.exists(path_to_inputs_to_final_model_folder):
     os.makedirs(path_to_inputs_to_final_model_folder)
###############################################################################
# Read the configuration dictionaries
configHolder = ConfigHolder.ConfigurationHolder.PrepareConfigHolderWithOutputFolderToAddBase(KVAEconfigurationPath, baseFolderPath)
config       = configHolder.config
###############################################################################
sequenceLength = config['sequence_length']
###############################################################################
# Loop over training, validation and testing
for dataCase in range(3):
    print(dataCase)
    ###########################################################################
    # Writing the location of the other data from there. These are fixed locations
    # starting from the base folder.
    # Each is saved in two files: one with also dimensions, and one without
    if dataCase == 0:
        path_to_images_folder = baseFolderPath + '/train_images/' # 1) Images
        path_to_GSs = baseFolderPath + '/train_GSs/train_GSs.mat'
        path_to_GSs_cells = baseFolderPath + '/train_GSs/train_GSs_cells.mat' # 2) Odometry
        path_to_inputs_to_final_model_full = path_to_inputs_to_final_model_folder + '/train_data_file_sl_{}_d1_{}_d2_{}'.format(
            sequenceLength, config['dimension_x'], config['dimension_y'])
        path_to_inputs_to_final_model = path_to_inputs_to_final_model_folder + '/train_data_file'
    elif dataCase == 1:
        path_to_images_folder = baseFolderPath + '/validation_images/'
        path_to_GSs = baseFolderPath + '/validation_GSs/validation_GSs.mat'
        path_to_GSs_cells = baseFolderPath + '/validation_GSs/validation_GSs_cells.mat'
        path_to_inputs_to_final_model_full = path_to_inputs_to_final_model_folder + '/validation_data_file_sl_{}_d1_{}_d2_{}'.format(
            sequenceLength, config['dimension_x'], config['dimension_y'])
        path_to_inputs_to_final_model = path_to_inputs_to_final_model_folder + '/validation_data_file'
    elif dataCase == 2:
        path_to_images_folder = baseFolderPath + '/test_images/'
        path_to_GSs = baseFolderPath + '/test_GSs/test_GSs.mat'
        path_to_GSs_cells = baseFolderPath + '/test_GSs/test_GSs_cells.mat'
        path_to_acceleration = baseFolderPath + '/test_IMU/accelerationPM.mat'
        path_to_angularVelocity = baseFolderPath + '/test_IMU/angularVelocityPM.mat'
        path_to_inputs_to_final_model_full = path_to_inputs_to_final_model_folder + '/test_data_file_sl_{}_d1_{}_d2_{}'.format(
            sequenceLength, config['dimension_x'], config['dimension_y'])
        path_to_inputs_to_final_model = path_to_inputs_to_final_model_folder + '/test_data_file'
    ###########################################################################
    # Extract data without including IMU (acceleration and angular velocity)
    #EDU.ExtractDataForKVAE(config, path_to_GSs, path_to_GSs_cells, path_to_images_folder, 
    #                       path_to_inputs_to_final_model, path_to_inputs_to_final_model_full, check_mode)

    # Extract data including IMU (acceleration and angular velocity)
    EDU.ExtractDataForKVAE(config, path_to_GSs, path_to_GSs_cells, path_to_images_folder, path_to_acceleration, path_to_angularVelocity,
                           path_to_inputs_to_final_model, path_to_inputs_to_final_model_full, check_mode)
