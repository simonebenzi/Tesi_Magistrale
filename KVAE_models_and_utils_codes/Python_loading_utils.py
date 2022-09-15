# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:30:40 2022

@author: asus
"""

from KVAE_models_and_utils_codes import ConfigurationHolder as ConfigHolder
from mat4py import loadmat
import numpy as np
import random

def ExtractBaseFolderPath(pathWhereDatasetFolderIsDefined):
    
    with open(pathWhereDatasetFolderIsDefined) as f:
        baseFolderPath = f.readlines()[0]
        baseFolderPath = baseFolderPath.rstrip() # Do this or in Ubuntu a \n is added    
    return baseFolderPath

def RedefinePathsVAEAddingBaseFolderPath(configHolder, baseFolderPath):
    
    configHolder.RedefinePathAddingBaseFolder('training_data_file', baseFolderPath)
    configHolder.RedefinePathAddingBaseFolder('validation_data_file', baseFolderPath)
    configHolder.RedefinePathAddingBaseFolder('test_data_file', baseFolderPath)
    configHolder.RedefinePathAddingBaseFolder('output_folder_kvae', baseFolderPath)
    return configHolder

def RedefinePathsKVAEAddingBaseFolderPath(configHolderKVAE, baseFolderPath):
    
    configHolderKVAE.RedefinePathAddingBaseFolder('minFile', baseFolderPath)
    configHolderKVAE.RedefinePathAddingBaseFolder('maxFile', baseFolderPath)
    configHolderKVAE.RedefinePathAddingBaseFolder('training_data_file', baseFolderPath)
    configHolderKVAE.RedefinePathAddingBaseFolder('validation_data_file', baseFolderPath)
    configHolderKVAE.RedefinePathAddingBaseFolder('testing_data_file', baseFolderPath)
    configHolderKVAE.RedefinePathAddingBaseFolder('clustering_data_file', baseFolderPath)
    return configHolderKVAE

def RedefinePathsCombinedMJPFsAddingBaseFolderPath(configHolderCombinedMJPFs, baseFolderPath):
    
    configHolderCombinedMJPFs.RedefinePathAddingBaseFolder('clustering_data_file_odom', baseFolderPath)
    configHolderCombinedMJPFs.RedefinePathAddingBaseFolder('clustering_data_file_video', baseFolderPath)
    configHolderCombinedMJPFs.RedefinePathAddingBaseFolder('kvae_file', baseFolderPath)
    configHolderCombinedMJPFs.RedefinePathAddingBaseFolder('reconstructedImagesFolder', baseFolderPath)    
    return configHolderCombinedMJPFs

def LoadMinAndMaxPositionalDataInConfig(config):
    
    minTrainingPositions = loadmat(config['minFile'])
    maxTrainingPositions = loadmat(config['maxFile'])
    config['minXReal'] = minTrainingPositions['dataMin'][0]
    config['minYReal'] = minTrainingPositions['dataMin'][1]
    config['maxXReal'] = maxTrainingPositions['dataMax'][0]
    config['maxYReal'] = maxTrainingPositions['dataMax'][1]    
    return config

def AddInitialTimeInstantsArrayToConfigForMultipleTests(configCombinedMJPFs, configCombinedMJPFsMultipleAttempts, testingData):
    
    numberOfTimeInstants = testingData.params.shape[0]*testingData.params.shape[1]
    numberOfAdditionalInitialTimeInstants = int(np.floor(numberOfTimeInstants*configCombinedMJPFs['newInitialTimeInstantsFrequency']))
    additionalInitialTimeInstants = random.sample(range(0, numberOfTimeInstants), numberOfAdditionalInitialTimeInstants)
    additionalInitialTimeInstants = np.asarray(additionalInitialTimeInstants)
    additionalInitialTimeInstants = np.append(additionalInitialTimeInstants, 0)
    configCombinedMJPFsMultipleAttempts['initialTimeInstant'] = additionalInitialTimeInstants
    return configCombinedMJPFsMultipleAttempts

def PrepareConfigsOfTrackingWithSingleParameter(pathWhereDatasetFolderIsDefined, 
                                                KVAEconfigurationPath, combinedMJPFsConfigurationPath):
    
    # Extract base folder path
    baseFolderPath = ExtractBaseFolderPath(pathWhereDatasetFolderIsDefined)
    # Configuration holders
    configHolderKVAE = ConfigHolder.ConfigurationHolder.PrepareConfigHolderWithOutputFolderToAddBase(
            KVAEconfigurationPath, baseFolderPath)
    configHolderCombinedMJPFs = ConfigHolder.ConfigurationHolder.PrepareConfigHolderWithOutputFolderToAddBase(
            combinedMJPFsConfigurationPath, baseFolderPath)
    # -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
    # Redefine the configuration holders
    configHolderKVAE = RedefinePathsKVAEAddingBaseFolderPath(configHolderKVAE, baseFolderPath)
    configHolderKVAE.config['batch_size'] = 1 
    configHolderCombinedMJPFs = RedefinePathsCombinedMJPFsAddingBaseFolderPath(configHolderCombinedMJPFs, baseFolderPath)
    
    return baseFolderPath, configHolderKVAE, configHolderCombinedMJPFs

def PrepareConfigsOfTrackingWithMultipleParameters(pathWhereDatasetFolderIsDefined, 
                                                   KVAEconfigurationPath, combinedMJPFsConfigurationPath, 
                                                   combinedMJPFsMultipleAttemptsConfigurationPath):
    
    baseFolderPath, configHolderKVAE, configHolderCombinedMJPFs = PrepareConfigsOfTrackingWithSingleParameter(
            pathWhereDatasetFolderIsDefined, KVAEconfigurationPath, combinedMJPFsConfigurationPath)
    # -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
    # + Configuration holder of multiple attempts
    configHolderCombinedMJPFsMultipleAttempts = ConfigHolder.ConfigurationHolder.PrepareConfigHolderWithOutputFolderToAddBase(
            combinedMJPFsMultipleAttemptsConfigurationPath, baseFolderPath)

    return configHolderKVAE, configHolderCombinedMJPFs, configHolderCombinedMJPFsMultipleAttempts
