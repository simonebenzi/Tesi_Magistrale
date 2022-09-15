
# Train the KVAE
###############################################################################
from KVAE_models_and_utils_codes import Train_KVAE
from KVAE_models_and_utils_codes import ConfigurationHolder as ConfigHolder
from KVAE_models_and_utils_codes import Python_loading_utils as PLU
###############################################################################
# Folders definition
# Path to configuration file of KVAE
KVAEconfigurationPath  = 'ConfigurationFiles/Config_KVAE.json'
# Path to .txt file containing the path to the dataset folder
pathWhereDatasetFolderIsDefined = 'ConfigurationFiles/BaseDataFolder.txt'
baseFolderPath = PLU.ExtractBaseFolderPath(pathWhereDatasetFolderIsDefined)
###############################################################################
configHolderKVAE = ConfigHolder.ConfigurationHolder.PrepareConfigHolderWithOutputFolderToAddBase(KVAEconfigurationPath, baseFolderPath)
###############################################################################
# Define the other paths
configHolderKVAE = PLU.RedefinePathsKVAEAddingBaseFolderPath(configHolderKVAE, baseFolderPath)
config = configHolderKVAE.config
###############################################################################
# Load min and max for positional data
config = PLU.LoadMinAndMaxPositionalDataInConfig(config)
###############################################################################
# TRAIN
kvae, summaryTrainingAllEpochs, summaryTrainingCurrentEpoch, summaryTestingAllEpochs, summaryTestingCurrentEpoch = \
   Train_KVAE.TrainWithValidationGivenLoadedConfiguration(config) 
# kvae, summaryTrainingAllEpochs, summaryTrainingCurrentEpoch = Train_KVAE.trainGivenLoadedConfiguration(config) 
#Train_KVAE.test(config, kvae) 