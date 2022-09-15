
###############################################################################
from KVAE_models_and_utils_codes import Combined_MJPFs_running_functions
from KVAE_models_and_utils_codes import Python_loading_utils as PLU
import numpy as np
###############################################################################
# PATHS
pathWhereDatasetFolderIsDefined = 'ConfigurationFiles/BaseDataFolder.txt'
KVAEconfigurationPath = 'ConfigurationFiles/Config_KVAE.json'
combinedMJPFsConfigurationPath = 'ConfigurationFiles/Config_KVAE_combined.json'
combinedMJPFsMultipleAttemptsConfigurationPath = 'ConfigurationFiles/Config_KVAE_combined_multiple_thresholds.json'
###############################################################################
# Prepare the configuration holders
configHolderKVAE, configHolderCombinedMJPFs, configHolderCombinedMJPFsMultipleAttempts = PLU.PrepareConfigsOfTrackingWithMultipleParameters(
        pathWhereDatasetFolderIsDefined, KVAEconfigurationPath, combinedMJPFsConfigurationPath, 
        combinedMJPFsMultipleAttemptsConfigurationPath)
configKVAE = configHolderKVAE.config
configCombinedMJPFs = configHolderCombinedMJPFs.config
configCombinedMJPFsMultipleAttempts = configHolderCombinedMJPFsMultipleAttempts.config
###############################################################################
# Define elements of the particular case
# Testing done on validation!!
configKVAE['testing_data_file'] = configKVAE['validation_data_file'] 
# We use particles restarting!!
configCombinedMJPFs['usingAnomalyThresholds'] = True
###############################################################################
# LOADING DATA FOR TRAINING AND TESTING
testingData = Combined_MJPFs_running_functions.PrepareTestDataForOdometryFromVideo(configKVAE, configCombinedMJPFs)
###############################################################################
# Adding initial time instants on validation data, to restart the processing
if configCombinedMJPFs['addingInitialTimeInstantsOnVal'] == True:
    configCombinedMJPFsMultipleAttempts = PLU.AddInitialTimeInstantsArrayToConfigForMultipleTests(
        configCombinedMJPFs, configCombinedMJPFsMultipleAttempts, testingData)
else:
    configCombinedMJPFsMultipleAttempts['initialTimeInstant'] = np.array([0])
###############################################################################
# PREPARE THE MODELS
kvaeOfV, transitionMatNumpy, transMatsTimeNumpy, nodesDistanceMatrixNumpy, \
    windowedtransMatsTimeNumpy = Combined_MJPFs_running_functions.PrepareModelsForOdometryFromVideoTestingData(
            configKVAE, configCombinedMJPFs, testingData)
###############################################################################
# MAIN LOOP
trackingCase = 111
kvaeOfV = Combined_MJPFs_running_functions.PerformMultipleCombinedMJPFsRunsWithStandardParametersGrid(
        configKVAE, configCombinedMJPFs, configCombinedMJPFsMultipleAttempts, trackingCase, kvaeOfV, testingData)
