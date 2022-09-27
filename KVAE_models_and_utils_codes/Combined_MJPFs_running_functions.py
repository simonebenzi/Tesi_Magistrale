# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:42:05 2021

@author: giulia.slavic
"""

import numpy as np
import torch
import copy
import scipy.io as sio

from KVAE_models_and_utils_codes import KVAE_D 
from KVAE_models_and_utils_codes import ReshapingData_utils as RD
from KVAE_models_and_utils_codes import DataHolder          as DH
from KVAE_models_and_utils_codes import ClusteringGraph     as CG
from KVAE_models_and_utils_codes import TestsGrid                          as TG 
from ConfigurationFiles          import Config_GPU          as ConfigGPUCode

from KVAE_models_and_utils_codes import KVAE_odometry_from_video          as KVAE_OfV
from KVAE_models_and_utils_codes import MarkovJumpParticleFilterOdometric as MJPFO
from KVAE_models_and_utils_codes import MarkovJumpParticleFilterVideo     as MJPFV

###############################################################################
# GPU or CPU?
configGPU = ConfigGPUCode.ConfigureGPUSettings()
device    = ConfigGPUCode.DefineDeviceVariable(configGPU)

##############################################################################
# Functions for saving outputs (general)

def SaveOutputTorchToMATLAB(currentValue, listOfValues, outputFolder, fileName, matlabVariableName):
    
    currentValue_numpy = RD.bringToNumpy(currentValue)
    SaveOutputNumpyToMATLAB(currentValue_numpy, listOfValues, outputFolder, fileName, matlabVariableName)
    
    return

def SaveOutputNumpyToMATLAB(currentValue, listOfValues, outputFolder, fileName, matlabVariableName):
    
    currentValueCopy = copy.deepcopy(currentValue)
    listOfValues.append(currentValueCopy)
    sio.savemat(outputFolder + '/' + fileName + '.mat', {matlabVariableName: listOfValues})
    
    return

def SaveOutputToMATLAB(currentValue, listOfValues, outputFolder, fileName, matlabVariableName):
    
    if type(currentValue)   == np.ndarray:
        SaveOutputNumpyToMATLAB(currentValue, listOfValues, outputFolder, fileName, matlabVariableName)
    elif type(currentValue) == torch.Tensor:
        SaveOutputTorchToMATLAB(currentValue, listOfValues, outputFolder, fileName, matlabVariableName)
        
    return

def SaveOutputToMATLABGivenDebugCode(currentValue, listOfValues, outputFolder, matlabVariableName, debugCode):
    
    fileName = 'OD_' + matlabVariableName + '_debugCode' + str(debugCode)
    SaveOutputToMATLAB(currentValue, listOfValues, outputFolder, fileName, matlabVariableName)
    
    return

def PrepareTrainDataForOdometryFromVideo(config, configOfV):
    
    # Training data
    trainingData = DH.DataHolder(dataFile = config['training_data_file']) 
    trainingData.Print()
    
    return trainingData

def PrepareTestDataForOdometryFromVideo(config, configOfV):
    
    # Testing data
    if configOfV['testingOnTraining'] == True:
        testingData  = DH.DataHolder(dataFile = config['training_data_file']) 
    else:
        testingData  = DH.DataHolder(dataFile = config['testing_data_file']) 
    testingData.Print()
    
    return testingData

def PrepareDataForOdometryFromVideo(config, configOfV):
    
    trainingData = PrepareTrainDataForOdometryFromVideo(config, configOfV)
    testingData = PrepareTestDataForOdometryFromVideo(config, configOfV)
    
    return trainingData, testingData
           
def PrepareModelsForOdometryFromVideoTestingData(config, configOfV, testingData):
    
    ###############################################################################
    # LOADING THE CLUSTERING GRAPH 

    clusterGraphParams, vectorOfKeptClusters = CG.ClusteringGraph.PrepareClusterGraphFromConfigFile(config)

    # Video cluster over a
    clusterGraphVideo, vectorOfKeptClusters  = CG.ClusteringGraph.PrepareClusterGraphFromConfigFileAndClusteringFile(config = config, 
                                                                                               clusteringFile = configOfV["clustering_data_file_video"])
    # Video cluster over z
    clusterGraphVideoZ, vectorOfKeptClusters = CG.ClusteringGraph.PrepareClusterGraphFromConfigFileAndClusteringFile(config = config, 
                                                                                               clusteringFile = configOfV["clustering_data_file_video"],
                                                                                               nodesMeanName  = 'nodesMeanstate', 
                                                                                               nodesCovName   = 'nodesCovstate')    
    # Numpy extraction for debugging
    transitionMatNumpy         = clusterGraphVideo.transitionMat.copy()
    transMatsTimeNumpy         = clusterGraphVideo.transMatsTime.copy()
    nodesDistanceMatrixNumpy   = clusterGraphVideo.nodesDistanceMatrix.copy()
    windowedtransMatsTimeNumpy = clusterGraphVideo.windowedtransMatsTime.copy()

    ###############################################################################
    # Initializing the KVAE
    kvae    = KVAE_D.KalmanVariationalAutoencoder_D(config          = config, 
                                                    clusterGraph    = clusterGraphParams, 
                                                    trainingData    = testingData,
                                                    sequence_length = 1).to(device)
    kvae.print()
    # Loading the KVAE_D parameters
    kvae.LoadTrainedModel(loadFile = configOfV['kvae_file'])
    kvae.eval()

    # Bring clusters to torch
    clusterGraphVideo.BringGraphToTorch()
    clusterGraphVideoZ.BringGraphToTorch()
    clusterGraphParams.BringGraphToTorch()

    ###############################################################################
    # Bring dataHolder data to torch 

    # Do this AFTER creating KVAE, so that cluster distances can be calculated with numpy
    testingData.BringDataToTorch()

    # Unroll data
    testingData.UnrollData(image_channels = config['image_channels'])

    ###############################################################################
    # CREATING THE KVAE filter odometry from video 

    obs_var            = configOfV['observationVariance']
    obs_cov            = (torch.eye(4)*obs_var).to(device)
    
    if hasattr(clusterGraphVideoZ, 'transitionMatExploration'):
        mjpf = MJPFO.MarkovJumpParticleFilterOdometric(numberOfParticles     = configOfV['N_Particles'], 
                                             dimensionOfState      = 4, 
                                             nodesMean             = clusterGraphParams.nodesMean, 
                                             nodesCov              = clusterGraphParams.nodesCov, 
                                             nodesCovPred          = clusterGraphVideo.nodesCovPred, 
                                             transitionMat         = clusterGraphParams.transitionMat, 
                                             observationCovariance = obs_cov,
                                             z_meaning             = config['z_meaning'], 
                                             transMatsTime         = clusterGraphParams.windowedtransMatsTime,
                                             maxClustersTime       = clusterGraphParams.maxClustersTime,
                                             vicinityTransitionMat = clusterGraphVideoZ.transitionMatExploration,
                                             resamplingThreshold   = configOfV['firstResampleThresh'], 
                                             observationTrustFactor = configOfV['obs'])
    else:
        mjpf = MJPFO.MarkovJumpParticleFilterOdometric(numberOfParticles     = configOfV['N_Particles'], 
                                             dimensionOfState      = 4, 
                                             nodesMean             = clusterGraphParams.nodesMean, 
                                             nodesCov              = clusterGraphParams.nodesCov, 
                                             nodesCovPred          = clusterGraphVideo.nodesCovPred, 
                                             transitionMat         = clusterGraphParams.transitionMat, 
                                             observationCovariance = obs_cov,
                                             z_meaning             = config['z_meaning'], 
                                             transMatsTime         = clusterGraphParams.windowedtransMatsTime,
                                             maxClustersTime       = clusterGraphParams.maxClustersTime,
                                             resamplingThreshold   = configOfV['firstResampleThresh'], 
                                             observationTrustFactor = configOfV['obs'])

    ###############################################################################
    # For mjpf on video, using the covariances from KVAE training
    
    noise_emission   = config['noise_emission']
    noise_transition = config['noise_transition']

    Q = noise_transition * torch.eye(config['dim_z']).to(device)
    R = noise_emission   * torch.eye(config['dim_a']).to(device)

    # Create a list of Qs
    Qs = torch.unsqueeze(Q,0)
    Qs = Qs.repeat(clusterGraphVideoZ.num_clusters,1,1)
    
    if hasattr(clusterGraphVideoZ, 'transitionMatExploration'):
        mjpf_video = MJPFV.MarkovJumpParticleFilterVideo(numberOfParticles     = configOfV['N_Particles'], 
                                             dimensionOfState      = config['dim_z'], 
                                             nodesMean             = clusterGraphVideoZ.nodesMean, 
                                             nodesCov              = Qs,#*10, # clusterGraphVideoZ.nodesCov, 
                                             transitionMat         = clusterGraphVideoZ.transitionMat, 
                                             observationCovariance = R,#,*100,  # obs_cov,
                                             transMatsTime         = clusterGraphVideoZ.windowedtransMatsTime,
                                             maxClustersTime       = clusterGraphVideoZ.maxClustersTime,
                                             kf                    = kvae.kf,
                                             vicinityTransitionMat = clusterGraphVideoZ.transitionMatExploration,
                                             resamplingThreshold   = configOfV['firstResampleThresh'])
    else:
        mjpf_video = MJPFV.MarkovJumpParticleFilterVideo(numberOfParticles     = configOfV['N_Particles'], 
                                             dimensionOfState      = config['dim_z'], 
                                             nodesMean             = clusterGraphVideoZ.nodesMean, 
                                             nodesCov              = Qs,#*10, # clusterGraphVideoZ.nodesCov, 
                                             transitionMat         = clusterGraphVideoZ.transitionMat, 
                                             observationCovariance = R,#/100,  # obs_cov,
                                             transMatsTime         = clusterGraphVideoZ.windowedtransMatsTime,
                                             maxClustersTime       = clusterGraphVideoZ.maxClustersTime,
                                             kf                    = kvae.kf,
                                             resamplingThreshold   = configOfV['firstResampleThresh'])
        
    # Means and stds of anomalies
    anomaliesMeans = [float(f) for f in configOfV['AnomaliesMeans'].split(',')]
    anomaliesMeans = np.asarray(anomaliesMeans)
    anomaliesStandardDeviations = [float(f) for f in configOfV['AnomaliesStandardDeviations'].split(',')]
    anomaliesStandardDeviations = np.asarray(anomaliesStandardDeviations)
    
    time_window = configOfV['time_window']
    time_enough_ratio = configOfV['time_enough_ratio']
    time_wait_ratio = configOfV['time_wait_ratio']
    stdTimes = configOfV['stdTimes']
    
    ###########################################################################
    kvaeOfV = KVAE_OfV.KVAE_odometry_from_video(kvae = kvae, 
                                                clusterGraphParams = clusterGraphParams, 
                                                clusterGraphVideo  = clusterGraphVideo, 
                                                clusterGraphVideoZ = clusterGraphVideoZ, 
                                                mjpf               = mjpf,
                                                mjpf_video         = mjpf_video,
                                                skew_video         = configOfV['skew_video'], 
                                                anomaliesMeans     = anomaliesMeans, 
                                                anomaliesStandardDeviations = anomaliesStandardDeviations,
                                                time_window        = time_window,
                                                time_enough_ratio  = time_enough_ratio,
                                                time_wait_ratio    = time_wait_ratio,
                                                stdTimes           = stdTimes,
                                                usingAnomalyThresholds = configOfV['usingAnomalyThresholds'],
                                                firstResampleThresh = configOfV['firstResampleThresh'],
                                                resampleThresh = configOfV['resampleThresh']).to(device)
    
    return kvaeOfV, transitionMatNumpy, transMatsTimeNumpy, nodesDistanceMatrixNumpy, windowedtransMatsTimeNumpy


def RunCombinedMJPF(config, configOfV, case, kvaeOfV, testingData, use_IMU):
    
    if case != 112: # if not speed calculation case
        newIndicesForSwapping_numpy_all = []
        indicesRestartedParticles_numpy_all = []
        alphas_numpy_all = []
        direct_alpha_from_video_numpy_all = []
        z_mus_update_numpy_all = []
        predicted_params_min_all = []
        clusterAssignments_numpy_all = []
        anomalies_odometry_numpy_all = []
        anomalies_hybrid_numpy_all = []
        allPredictedParams_numpy_all = []
        clusterAssignments_od_numpy_all = []
        odometryUpdatedParticles_od_numpy_all = []
        particlesWeights_numpy_all = []
        anomalies_numpy_all = []

    # Testing loop
    print('Filtering... please wait...')

    kvaeOfV.eval()
    
    if configOfV['lastTimeInstant'] == -1:
        lenghtLoop = testingData.params.shape[0]
    else: 
        lenghtLoop = configOfV['lastTimeInstant']
        
    with torch.no_grad():
        
        print('Beginning tracking.')
        print('Number of time instants: ' + str(lenghtLoop))
        
        # Looping over the batches of training data
        #for i in range(testingData.images.shape[0]):
        for i in range(int(configOfV['initialTimeInstant']), lenghtLoop):
            
            print('Data number ' + str(i+1) + ' out of ' + str(lenghtLoop))
            
            # Extract from data
            if use_IMU:
                currentImagesBatch, currentControlsBatch, currentOdometryBatch, currentParamsBatch, currentAccBatch, currentOrientBatch = \
                    kvaeOfV.kvae.ExtractBatchInputsDataStructure4DWithoutDistances(data = testingData, 
                                                                                currentBatchNumber = i, 
                                                                                batchSize = config['batch_size'], 
                                                                                image_channels = config['image_channels'], 
                                                                                use_IMU = use_IMU)
            else:
                currentImagesBatch, currentControlsBatch, currentOdometryBatch, currentParamsBatch = \
                    kvaeOfV.kvae.ExtractBatchInputsDataStructure4DWithoutDistances(data = testingData, 
                                                                                currentBatchNumber = i, 
                                                                                batchSize = config['batch_size'], 
                                                                                image_channels = config['image_channels'],
                                                                                use_IMU = use_IMU) 
            # If there is no color channel, add one after batch size
            if len(currentImagesBatch.shape) == 4:
                currentImagesBatch = torch.unsqueeze(currentImagesBatch, 1)
            if config['image_channels'] != 1 and currentImagesBatch.shape[2] != 1:
                currentImagesBatch = torch.swapaxes(currentImagesBatch, 1,2)
            if len(currentImagesBatch.shape) == 4:
                currentImagesBatch = torch.unsqueeze(currentImagesBatch, 1).to(device)            
            # If this is the first time instant of a trajectory, set the 'timeInstant' variable in the filter
            isThisAStartingPoint = i in testingData.startingPoints
            if i == int(configOfV['initialTimeInstant']) or isThisAStartingPoint:
                kvaeOfV.timeInstant = 0
                
            ######################### CALL TO KVAE ############################ <------------------------------------
            
            if case == 100:
                
                print('case 100')
        
                allPredictedParams, alpha, mu_t = \
                   kvaeOfV.CheckFindingOdometryFromZUsingZMatricesGivenOdometryAlphas(currentImagesBatch, currentParamsBatch, 
                                                                                      currentOdometryBatch)
                      
                SaveOutputToMATLABGivenDebugCode(alpha, alphas_numpy_all, configOfV['output_folder'], 'alpha', case)
                SaveOutputToMATLABGivenDebugCode(allPredictedParams, allPredictedParams_numpy_all, configOfV['output_folder'], 'predictedParams', case)
                SaveOutputToMATLABGivenDebugCode(mu_t, z_mus_update_numpy_all, configOfV['output_folder'], 'mus', case)
                
        
                currentParamsBatch_flattened = torch.squeeze(currentParamsBatch)
                error_min                    = 10000000000
                # Loopint over the number of cluster
                for i in range(allPredictedParams.shape[0]):
                    errors_current_cluster = torch.mean(torch.abs(currentParamsBatch_flattened - allPredictedParams[i,:]))
                    if error_min > errors_current_cluster:
                        predicted_params_min_error = allPredictedParams[i,:]
                        error_min = errors_current_cluster         
                SaveOutputToMATLABGivenDebugCode(predicted_params_min_error, predicted_params_min_all, 
                                                 configOfV['output_folder'], 'predicted_params_min_error', case)

            elif case == 110:
                
                # Different implementation using IMU or not
                if not use_IMU:
                    kvaeOfV.CombinedMJPFsVideoOdometrySingleMJPF(currentImagesBatch = currentImagesBatch,
                        currentParamsBatch = currentParamsBatch,
                        outputFolder = configOfV['output_folder'],
                        type_of_weighting = configOfV['type_of_weighting'],
                        knownStartingPoint = configOfV['known_starting_point'], 
                        saveReconstructedImages = configOfV['saveReconstructedImages'],
                        reconstructedImagesFolder = configOfV['reconstructedImagesFolder'],
                        fastDistanceCalculation = configOfV['fastDistanceCalculation'],
                        percentageParticlesToReinitialize = configOfV['percentageParticlesToReinitialize'])
                else:
                    kvaeOfV.CombinedMJPFsVideoOdometrySingleMJPFWithIMU(currentImagesBatch = currentImagesBatch,
                        currentParamsBatch = currentParamsBatch,
                        currentAccBatch = currentAccBatch,
                        currentOrientBatch = currentOrientBatch,
                        outputFolder = configOfV['output_folder'],
                        type_of_weighting = configOfV['type_of_weighting'],
                        knownStartingPoint = configOfV['known_starting_point'], 
                        saveReconstructedImages = configOfV['saveReconstructedImages'],
                        reconstructedImagesFolder = configOfV['reconstructedImagesFolder'],
                        fastDistanceCalculation = configOfV['fastDistanceCalculation'],
                        percentageParticlesToReinitialize = configOfV['percentageParticlesToReinitialize'])
                
                SaveOutputToMATLABGivenDebugCode(kvaeOfV.updatedValuesFromDMatrices.clone(), allPredictedParams_numpy_all, 
                                                 configOfV['output_folder'], 'predictedParams', case)
                
                SaveOutputToMATLABGivenDebugCode(kvaeOfV.alpha_from_video_plus_sequencing.clone(), alphas_numpy_all, 
                                                 configOfV['output_folder'], 'alpha', case)
                
                SaveOutputToMATLABGivenDebugCode(kvaeOfV.direct_alpha_from_video.clone(), direct_alpha_from_video_numpy_all, 
                                                 configOfV['output_folder'], 'direct_alpha_from_video', case)
                
                SaveOutputToMATLABGivenDebugCode(kvaeOfV.mjpf_video.clusterAssignments, clusterAssignments_numpy_all, 
                                                 configOfV['output_folder'], 'clusterAssignments', case)
                
                SaveOutputToMATLABGivenDebugCode(kvaeOfV.mjpf.clusterAssignments, clusterAssignments_od_numpy_all, 
                                                 configOfV['output_folder'], 'clusterAssignments_od', case)
                
                SaveOutputToMATLABGivenDebugCode(kvaeOfV.updatedOdometryValuesBeforeResampling.clone(), odometryUpdatedParticles_od_numpy_all, 
                                                 configOfV['output_folder'], 'odometryUpdatedParticles_od', case)
                
                SaveOutputToMATLABGivenDebugCode(kvaeOfV.mjpf_video.particlesWeights.clone(), particlesWeights_numpy_all, 
                                                 configOfV['output_folder'], 'particlesWeights', case)
                
                SaveOutputToMATLABGivenDebugCode(kvaeOfV.anomalies_odometry.clone(), anomalies_odometry_numpy_all, 
                                                 configOfV['output_folder'], 'anomalies_odometry', case)

                #print(kvaeOfV.timeInstant)
                #if kvaeOfV.timeInstant > 1:
                    #SaveOutputToMATLABGivenDebugCode(kvaeOfV.hybridUpdate.clone(), anomalies_hybrid_numpy_all, 
                                                    #configOfV['output_folder'], 'odometryHybridUpdatedParticles_od', case)
                
                whenRestarted_numpy = np.asarray(kvaeOfV.whenRestarted)
                sio.savemat(configOfV['output_folder'] + '/OD_whenRestarted_numpy_debugCode110' + '.mat', 
                            {'whenRestarted':whenRestarted_numpy})
                
                whyRestarted_numpy = np.asarray(kvaeOfV.whyRestarted)
                sio.savemat(configOfV['output_folder'] + '/OD_whyRestarted_numpy_debugCode110' + '.mat', 
                            {'whyRestarted':whyRestarted_numpy})
                 
                newIndicesForSwapping_numpy = kvaeOfV.newIndicesForSwapping
                newIndicesForSwapping_numpy_all.append(newIndicesForSwapping_numpy)
                sio.savemat(configOfV['output_folder'] + '/OD_newIndicesForSwapping_numpy_debugCode110' + '.mat', 
                            {'newIndicesForSwapping':newIndicesForSwapping_numpy_all})
                
                indicesRestartedParticles_numpy = kvaeOfV.indicesRestartedParticles
                indicesRestartedParticles_numpy_all.append(indicesRestartedParticles_numpy)
                sio.savemat(configOfV['output_folder'] + '/OD_indicesRestartedParticles_numpy_debugCode110' + '.mat', 
                            {'indicesRestartedParticles':indicesRestartedParticles_numpy_all})

                ###################################################################
                # ANOMALIES
                SaveOutputToMATLABGivenDebugCode(kvaeOfV.anomalyVectorCurrentTimeInstant, anomalies_numpy_all, 
                                                 configOfV['output_folder'], 'anomalies', case)
                    
            elif case == 111:

                kvaeOfV.CombinedMJPFsVideoOdometrySingleMJPF(currentImagesBatch = currentImagesBatch,
                                                             currentParamsBatch = currentParamsBatch,
                                                             outputFolder = configOfV['output_folder'],
                                                             type_of_weighting = configOfV['type_of_weighting'],
                                                             knownStartingPoint = configOfV['known_starting_point'], 
                                                             saveReconstructedImages = configOfV['saveReconstructedImages'],
                                                             reconstructedImagesFolder = configOfV['reconstructedImagesFolder'],
                                                             fastDistanceCalculation = configOfV['fastDistanceCalculation'],
                                                             percentageParticlesToReinitialize = configOfV['percentageParticlesToReinitialize'])

                SaveOutputToMATLABGivenDebugCode(kvaeOfV.updatedValuesFromDMatrices.clone(), allPredictedParams_numpy_all, 
                                                 configOfV['output_folder'], 'predictedParams', case-1)
                
                SaveOutputToMATLABGivenDebugCode(kvaeOfV.mjpf_video.clusterAssignments, clusterAssignments_numpy_all, 
                                                 configOfV['output_folder'], 'clusterAssignments', case-1)
                
                SaveOutputToMATLABGivenDebugCode(kvaeOfV.updatedOdometryValuesBeforeResampling.clone(), odometryUpdatedParticles_od_numpy_all, 
                                                 configOfV['output_folder'], 'odometryUpdatedParticles_od', case-1)
                
                SaveOutputToMATLABGivenDebugCode(kvaeOfV.mjpf_video.particlesWeights.clone(), particlesWeights_numpy_all, 
                                                 configOfV['output_folder'], 'particlesWeights', case-1)
                
                SaveOutputToMATLABGivenDebugCode(kvaeOfV.anomalies_odometry.clone(), anomalies_odometry_numpy_all, 
                                                 configOfV['output_folder'], 'anomalies_odometry', case-1)
                
                whenRestarted_numpy = np.asarray(kvaeOfV.whenRestarted)
                sio.savemat(configOfV['output_folder'] + '/OD_whenRestarted_numpy_debugCode110' + '.mat', 
                            {'whenRestarted':whenRestarted_numpy})
                
                whyRestarted_numpy = np.asarray(kvaeOfV.whyRestarted)
                sio.savemat(configOfV['output_folder'] + '/OD_whyRestarted_numpy_debugCode110' + '.mat', 
                            {'whyRestarted':whyRestarted_numpy})

                newIndicesForSwapping_numpy = kvaeOfV.newIndicesForSwapping
                newIndicesForSwapping_numpy_all.append(newIndicesForSwapping_numpy)
                sio.savemat(configOfV['output_folder'] + '/OD_newIndicesForSwapping_numpy_debugCode110' + '.mat', 
                            {'newIndicesForSwapping':newIndicesForSwapping_numpy_all})
                
                indicesRestartedParticles_numpy = kvaeOfV.indicesRestartedParticles
                indicesRestartedParticles_numpy_all.append(indicesRestartedParticles_numpy)
                sio.savemat(configOfV['output_folder'] + '/OD_indicesRestartPart_numpy_debugCode110' + '.mat', 
                            {'indicesRestartedParticles':indicesRestartedParticles_numpy_all})
                
            elif case == 112: # SPEED TESTING
                
                # fast version: no unnecessary lines to save information for plotting
                kvaeOfV.CombinedMJPFsVideoOdometrySingleMJPF(currentImagesBatch = currentImagesBatch,
                                                             currentParamsBatch = currentParamsBatch,
                                                             outputFolder = configOfV['output_folder'],
                                                             type_of_weighting = configOfV['type_of_weighting'],
                                                             knownStartingPoint = configOfV['known_starting_point'], 
                                                             saveReconstructedImages = configOfV['saveReconstructedImages'],
                                                             reconstructedImagesFolder = configOfV['reconstructedImagesFolder'],
                                                             fastDistanceCalculation = configOfV['fastDistanceCalculation'],
                                                             percentageParticlesToReinitialize = configOfV['percentageParticlesToReinitialize'])
                
                

            #######################################################################
            # Empty GPU from useless data            
            del currentImagesBatch
            del currentControlsBatch
            del currentOdometryBatch
            del currentParamsBatch            
            if device.type == "cuda":
                torch.cuda.empty_cache() 
    
    return kvaeOfV

###############################################################################
# CODES FOR RUNNING WITH MULTIPLE PARAMETERS

# Function to attempt running using a set of different parameters in order to
# later find which combination of parameters is the best one.
# INPUTS:
# - parametersGrid: a grid containing the values of the different parameters to 
#                 set. The number of rows corresponds to the number of 
#                 different attempts one wants to perform; the number of 
#                 columns to the number of parameters.
#                 Type: numpy 2D array.
# - parametersNames: the names of each parameter in the grid, given in the same
#                 sequence as in the grid columns.
#                 Type: numpy array or list.
# - config: configuration dictionary.
# - indexOfGridWhereToBegin: row index where to begin with performing the different
#                 training attempts. This allows to easily continue a training
#                 if it is stopped midway. 
#                 Type: int.
# OUTPUTS:
# None
def PerformMultipleCombinedMJPFsRuns(parametersGrid, config, configOfV, configMultipleParams, 
                                     case, kvaeOfV, testingData, indexOfGridWhereToBegin = 0):
    
    # The number of total experiments corresponds to the number of rows of the grid
    numberOfTotalExperiments = parametersGrid.numberOfTests        
    # Looping over the experiments
    for gridRow in range(indexOfGridWhereToBegin, numberOfTotalExperiments):
        
        configOfV_new, kvaeOfV = ChangeCombinedMJPFsParametersBasedOnGridRow(configOfV, configMultipleParams,
                                                                             parametersGrid, gridRow, kvaeOfV)
        # Perform the run 
        RunCombinedMJPF(config, configOfV_new, case, kvaeOfV, testingData)

    return

def ChangeCombinedMJPFsParametersBasedOnGridRow(configOfV, configMultipleParams, parametersGrid, gridRow, kvaeOfV):
    
    configOfV_new                  = parametersGrid.AssignToDictionaryTheValuesOfGridRow(configOfV, gridRow)
    configOfV_new                  = configOfV_new.copy()
    configOfV_new['output_folder'] = parametersGrid.DefineOutputFolderBasedOnRowOfGrid(configMultipleParams['output_folder'], gridRow)    
    kvaeOfV.ReassignParametersFromConfigHolder(configOfV_new)
    
    return configOfV_new, kvaeOfV

def PerformMultipleCombinedMJPFsRunsWithStandardParametersGrid(config, configOfV, configMultipleParams,
                                                               case, kvaeOfV, testingData, indexOfGridWhereToBegin = 0):
    
    # Create parameters grid
    parametersGrid = TG.TestsGrid.InitializeTestsGridFromConfigHolder(configMultipleParams)
    parametersGrid.SaveGridToMATLAB(configMultipleParams['output_folder'], 'parametersGrid')
    # Perform the multiple trainings
    PerformMultipleCombinedMJPFsRuns(parametersGrid, config, configOfV, configMultipleParams, 
                                     case, kvaeOfV, testingData, indexOfGridWhereToBegin)
    
    return