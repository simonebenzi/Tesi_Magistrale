
###############################################################################
from KVAE_models_and_utils_codes import Combined_MJPFs_running_functions
from KVAE_models_and_utils_codes import Python_loading_utils as PLU
###############################################################################
# PATHS
pathWhereDatasetFolderIsDefined = 'ConfigurationFiles/BaseDataFolder.txt'
KVAEconfigurationPath = 'ConfigurationFiles/Config_KVAE.json'
combinedMJPFsConfigurationPath = 'ConfigurationFiles/Config_KVAE_combined.json'
###############################################################################
# Prepare the configuration holders
baseFolderPath, configHolderKVAE, configHolderCombinedMJPFs = PLU.PrepareConfigsOfTrackingWithSingleParameter(
        pathWhereDatasetFolderIsDefined, KVAEconfigurationPath, combinedMJPFsConfigurationPath)
configKVAE = configHolderKVAE.config
configCombinedMJPFs = configHolderCombinedMJPFs.config
###############################################################################
# Define elements of the particular case
# Output folder to redefine
outputFolder = baseFolderPath + '/Tracking_results_training_best_parameters'
configHolderCombinedMJPFs.RedefineAndCreatesOutputFolder(outputFolder)
# Testing done on training!!
configKVAE['testing_data_file'] = configKVAE['training_data_file'] 
# Don't use anomaly threholds
configCombinedMJPFs['usingAnomalyThresholds'] = False
###############################################################################
# LOADING DATA FOR TRAINING AND TESTING
testingData = Combined_MJPFs_running_functions.PrepareTestDataForOdometryFromVideo(configKVAE, configCombinedMJPFs)
###############################################################################
# PREPARE THE MODELS
kvaeOfV, transitionMatNumpy, transMatsTimeNumpy, nodesDistanceMatrixNumpy, \
    windowedtransMatsTimeNumpy = Combined_MJPFs_running_functions.PrepareModelsForOdometryFromVideoTestingData(
            configKVAE, configCombinedMJPFs, testingData)
###############################################################################
# MAIN LOOP
trackingCase = 110
kvaeOfV = Combined_MJPFs_running_functions.RunCombinedMJPF(configKVAE, configCombinedMJPFs, trackingCase, kvaeOfV, testingData, False)
