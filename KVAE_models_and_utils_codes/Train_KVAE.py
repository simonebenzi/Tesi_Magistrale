

###############################################################################
# Import libraries
import os
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from matplotlib import figure
###############################################################################
# Import custom libraries/codes
from KVAE_models_and_utils_codes import ConfigurationHolder                as ConfigHolder
from ConfigurationFiles          import Config_GPU                         as ConfigGPU
from KVAE_models_and_utils_codes import DataLoader                         as DL
from KVAE_models_and_utils_codes import ClusteringGraph                    as CG
from KVAE_models_and_utils_codes import PlotGraphs_utils                   as PG
from KVAE_models_and_utils_codes import SummaryHolder                      as SH
from KVAE_models_and_utils_codes import SummaryHolderLossesAcrossEpochs    as SHLAE
from KVAE_models_and_utils_codes import ReshapingData_utils                as RD
from KVAE_models_and_utils_codes import TestsGrid                          as TG 
from KVAE_models_and_utils_codes import Distance_utils  as d_utils
#from KVAE_models_and_utils_codes import GradientFlowChecker                as GFC 
from KVAE_models_and_utils_codes import VAE
from KVAE_models_and_utils_codes import KVAE
from KVAE_models_and_utils_codes import KVAE_D  

import unittest

###############################################################################
# ---------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)

print('DEVICE USED:')
print(device)

###############################################################################
# Defining the names of the losses and other variables that we want to save
# in the summary holders.
# This contains only those value of which we are interested in the MEAN over all epochs
summaryNamesAllEpochs    = ['VAE_losses', 'Reconstruction_losses', 'KLD_losses', 'd_losses', 'd_losses_denorm', 
                            'd_loss_denorm_winning', 'Emission_losses', 'Transition_losses', 'Total_loss', 
                            'Prediction_error', 'No_motion_error', 'smoothingFilteringEqualityLoss',
                            'mean_of_alpha_max', 'tripletLossZ']
# This instead, contains values that are only epoch specific
summaryNamesCurrentEpoch = summaryNamesAllEpochs + ['As', 'Bs', 'Cs', 'a_states', 'z_states', 
                                                    'learning_rates', 'alphas', 'z_states_pred',
                                                    'z_sigmas_pred', 'odometry', 'real_params',
                                                    'predicted_params_all', 'predicted_params_min']

###############################################################################
# Single functions to load the different parts necessary for VAE/KVAE training.

def LoadTrainAndValidationInfo(config):
    
    # ----------------- LOADING DATA FOR TRAINING AND TESTING -----------------
    shuffledTrainingData, newIndicesShuffled, shuffledValidationData, newIndicesShuffledValidation = \
       DL.LoadTrainingAndValidationDataForVAE(config)       
    config, trainingData, newIndices, inverseIndexing, validationData, \
       newIndicesValidation, inverseIndexingValidation = DL.LoadTrainingAndValidationDataForKVAE(config)
       
    # --------------------- LOADING THE CLUSTERING GRAPH ----------------------
    print('Load clustering graph')
    clusterGraph, vectorOfKeptClusters = CG.ClusteringGraph.PrepareClusterGraphFromConfigFile(config)
    print('Save additional clustering info')
    # Saving if a cluster is saved or not
    sio.savemat(config['output_folder'] + '/vectorOfKeptOdometryClusters' + '.mat', {'vectorOfKeptOdometryClusters':vectorOfKeptClusters})
    # Save new cluster sequencing
    sio.savemat(config['output_folder'] + '/newClustersSequence' + '.mat', {'newClustersSequence':clusterGraph.clustersSequence})

    # Number of batches in the training data
    numberOfDataBatches     = trainingData.sequences // config['batch_size'] # takes floor value
    # Number of batches in the testing data
    numberOfDataBatchesValidation = validationData.sequences // config['batch_size_test'] # takes floor value
    
    return shuffledTrainingData, newIndicesShuffled, shuffledValidationData, newIndicesShuffledValidation, \
           config, trainingData, newIndices, inverseIndexing, validationData, newIndicesValidation, \
           inverseIndexingValidation, clusterGraph, numberOfDataBatches, numberOfDataBatchesValidation
    
# Function to initialize the KVAE
# INPUTS:
# - config: configuration dictionary, 
# - clusterGraph: clustering object
# - trainingData: data of training, to calculate distances of each point from the
#               clusters of the clusteringGraph
# OUTPUTS:
# - kvae: inizialized kvae
def InitializeKVAE(config, clusterGraph, trainingData):

    # ------------------------ CREATING THE KVAE-------------------------------    
    # Create a KVAE
    kvae    = KVAE.KalmanVariationalAutoencoder(config       = config, 
                                                clusterGraph = clusterGraph, 
                                                trainingData = trainingData).to(device)
    
    kvae.print()
    
    return kvae

# Function to initialize the KVAE
# INPUTS:
# - config: configuration dictionary, 
# - clusterGraph: clustering object
# - trainingData: data of training, to calculate distances of each point from the
#               clusters of the clusteringGraph
# OUTPUTS:
# - kvae: inizialized kvae
def InitializeKVAE_D(config, clusterGraph, trainingData):
    
    kvae    = KVAE_D.KalmanVariationalAutoencoder_D(config       = config, 
                                                    clusterGraph = clusterGraph, 
                                                    trainingData = trainingData).to(device)

    kvae.print()
    
    return kvae

# Function to bring all the objects necessary in training from numpy to torch
# INPUTS:
# - trainingData: training data in sequential order (for KVAE)
# - shuffledTrainingData: shuffled training data (for VAE)
# - testingData: testing data in sequential order (for KVAE)
# - shuffledTestingData: shuffled testing data (for VAE)
# - clusterGraph: clustering object
# OUTPUTS:
# The same as the inputs, but now as torch arrays.
def BringTrainingTestingDataToTorch(trainingData, shuffledTrainingData, testingData, 
                           shuffledTestingData):
    
    trainingData.BringDataToTorch()
    shuffledTrainingData.BringDataToTorch()
    testingData.BringDataToTorch()
    shuffledTestingData.BringDataToTorch()
    
    return trainingData, shuffledTrainingData, testingData, shuffledTestingData

###############################################################################
# Overall training PREPARATION function.

# Function that prepares all the necessary parts for performing KVAE training.
# INPUTS:
# - config: loaded configuration dictionary.
# OUTPUTS:
# - shuffled training and testing data for VAE training with indices to go 
#   back to unshuffled sequencing;
# - unshuffled training and testing data for KVAE training, with indices to go 
#   back to original sequencing, if the sequences were put in different order
#   for better KVAE training.
# - cluster graph.
# - number of batches for training and testing.
# - initialized kvae with all its subcomponents.
def BringAllObjectsToTorch(trainingData, shuffledTrainingData, testingData, 
                           shuffledTestingData, clusterGraph, trainingDistances, testingDistances):
    
    # ------------------------ Bring dataHolder data to torch -----------------    
    # Do this AFTER creating KVAE, so that cluster distances can be calculated
    # with numpy
    trainingData, shuffledTrainingData, testingData, shuffledTestingData = BringTrainingTestingDataToTorch(
        trainingData, shuffledTrainingData, testingData, shuffledTestingData)
    
    clusterGraph.BringGraphToTorch()
    
    trainingDistances = torch.from_numpy(trainingDistances).to(device)
    testingDistances  = torch.from_numpy(testingDistances).to(device)
    
    return trainingData, shuffledTrainingData, testingData, shuffledTestingData, clusterGraph, \
           trainingDistances, testingDistances

###############################################################################
# Overall training PREPARATION function.

# Function that prepares all the necessary parts for performing KVAE training.
# INPUTS:
# - config: loaded configuration dictionary.
# OUTPUTS:
# - shuffled training and testing data for VAE training with indices to go 
#   back to unshuffled sequencing;
# - unshuffled training and testing data for KVAE training, with indices to go 
#   back to original sequencing, if the sequences were put in different order
#   for better KVAE training.
# - cluster graph.
# - number of batches for training and testing.
# - initialized kvae with all its subcomponents.
def PrepareEverythingForTrainingAndValidation(config):
    
    # Load the training data
    shuffledTrainingData, newIndicesShuffled, shuffledTestingData, newIndicesShuffledTesting, \
       config, trainingData, newIndices, inverseIndexing, testingData, newIndicesTesting, \
       inverseIndexingTesting, clusterGraph, numberOfDataBatches, numberOfDataBatchesTest = LoadTrainAndValidationInfo(config)
       
    print('Initializing the KVAE')
       
    # If KVAE with odometry from video
    kvae = InitializeKVAE_D(config, clusterGraph, trainingData)
    
    # CALCULATING THE DISTANCES FROM THE CLUSTERS 
    # Calculating it only once at the beginning, so no recalculation is necessary at every loop.
    trainingDistances = kvae.FindDistancesFromClusters(trainingData.params)    
    testingDistances  = kvae.FindDistancesFromClusters(testingData.params)
        
    # Bring the objects to torch
    trainingData, shuffledTrainingData, testingData, shuffledTestingData, clusterGraph, \
       trainingDistances, testingDistances = \
       BringAllObjectsToTorch(trainingData, shuffledTrainingData, testingData, 
                              shuffledTestingData, clusterGraph, trainingDistances, testingDistances)  

    return shuffledTrainingData, newIndicesShuffled, shuffledTestingData, newIndicesShuffledTesting, \
           trainingData, newIndices, inverseIndexing, testingData, newIndicesTesting, \
           inverseIndexingTesting, clusterGraph, numberOfDataBatches, numberOfDataBatchesTest, \
           trainingDistances, testingDistances, kvae

###############################################################################
# TRAINING AND TESTING LOOP over VAE (or loading already trained VAE).

# Function to train and test the VAE only (so subpart of KVAE).
# If the VAE was already trained, it is loaded from the path described in 
# 'config' dictionary, otherwise it is trained from scratch.
# INPUTS:
# - kvae object, as the VAE is a subpart of it;
# - confi: configuration dictionary
# - shuffledTrainingData: training data, shuffled
# - shuffledTestingData: testing data, shuffled
def PerformVAETrainingAndTesting(vae, config, shuffledTrainingData, shuffledTestingData):
    
    print('VAE training/testing beginning...')

    # ------------------ PERFORM THE VAE MAIN LOOP ON TRAIN DATA --------------    
    # Initial learning rate
    learningRate = config['lr_only_vae']
    
    if config['image_channels'] != 1:
        trainingImages = torch.swapaxes(shuffledTrainingData.images, 0, 1)
        testingImages  = torch.swapaxes(shuffledTestingData.images, 0, 1)
    else:
        trainingImages = shuffledTrainingData.images
        testingImages  = shuffledTestingData.images
    
    # Loading if VAE was already trained ...
    if config['VAE_already_trained'] == True:    
        # Search first for the file in the config['output_folder'] folder, and if it
        # is not there, look in the parent directory
        vae_file_path = config['output_folder'] + config['trained_VAE_file_name']
        if os.path.isfile(vae_file_path) == False:
            vae_file_path = os.path.dirname(os.path.dirname(vae_file_path))
            vae_file_path = vae_file_path + '/' + config['trained_VAE_file_name'] 
        print('File path of trained VAE:')
        print(vae_file_path)
        # load vae    
        vae.load_state_dict(torch.load(vae_file_path))
        vae.to(device)           
        # Summary names
        summaryNamesAllEpochsVAE = VAE.VAE.ReturnSummaryNamesAllEpochsVAE()
        # Initialize summary objects over training and testing
        summaryTrainingAllEpochs = SH.SummaryHolder(summaryNamesAllEpochs)
        summaryTestingAllEpochs  = SH.SummaryHolder(summaryNamesAllEpochsVAE)
        
        summaryTrainingCurrentEpoch = vae.TestVAEOverEpoch(testingImages = trainingImages, 
                                                          outputFolder = config['output_folder'],
                                                          currentEpochNumber = config['only_vae_epochs']-1, 
                                                          batchSize = config['batch_size_VAE'],
                                                          alpha = config['alpha_VAEonly_training'])
        summaryTestingCurrentEpoch  = vae.TestVAEOverEpoch(testingImages = testingImages, 
                                                          outputFolder = config['output_folder'],
                                                          currentEpochNumber = config['only_vae_epochs']-1, 
                                                          batchSize = config['batch_size_VAE'],
                                                          alpha = config['alpha_VAEonly_training'])    
    # ... otherwise train the VAE.        
    else:  
        summaryTrainingAllEpochs, summaryTestingAllEpochs, summaryTrainingCurrentEpoch, \
            summaryTestingCurrentEpoch, learningRate = vae.TrainAndTestVAE(
            trainingImages = trainingImages, testingImages = testingImages,
            outputFolder = config['output_folder'],
            epochs = config['only_vae_epochs'], batchSize = config['batch_size_VAE'],
            decayRate = config['decay_rate'], decaySteps = config['decay_steps'], 
            alpha = config['alpha_VAEonly_training'], learningRate = learningRate, 
            weightDecay = config['weight_decay'], maxGradNorm = 300, optimizer = 'Adam')
        
    vae.FreezeBatchNormLayers()
    
    return vae, summaryTrainingAllEpochs, summaryTestingAllEpochs, summaryTrainingCurrentEpoch, summaryTestingCurrentEpoch

# Load the KVAE if we have a KVAE that was trained for both VAE and KF (without
# the training of the two parts in common).
# INPUTS:
# - kvae empty
# - config: configuration dictionary
def LoadKVAEwithTrainedKF(kvae, config):
    
    # --------------- If KVAE was already trained up to the matrices ----------    
    # In case the kf matrices had already been trained: 
    if config['KVAE_only_kf_already_trained'] == True:
        beginEpoch = config['epoch_of_already_trained_kf']
        kvae.load_state_dict(torch.load(config['output_folder'] + config['trained_KVAE_file_name']))
        #kvae.baseVAE.FreezeBatchNormLayers()
    else:
        beginEpoch = config['only_vae_epochs']
    
    return kvae, beginEpoch

###############################################################################
# Functions for TRAINING/TESTING of KVAE.
    
# Function to compare the values of the prediction with the case in which we
# keep the same value of the previous time instant (so we do not move).
# Obviously, we want for the prediction of our model to be better of just 
# predicting to not move, so keep an eye on the output of this.
def ComparePredictionsVsNoMotion(latentStateZ, predictedLatentStateZ, correctLatentStateZ, kvae):
    
    #######################################################################
    # Compare prediction against no prediction
    # Compare that predicting with matrix A is better than just picking the 
    # previous value
    previous_values  = latentStateZ[:,:-1,:]
    previous_values_reshaped = torch.reshape(previous_values, [-1, kvae.kf.dim_z])
    
    error_predictions = predictedLatentStateZ - correctLatentStateZ
    error_predictions = torch.abs(error_predictions)
    error_predictions = torch.mean(torch.mean(error_predictions))
    
    error_noMotions   = previous_values_reshaped - correctLatentStateZ
    error_noMotions   = torch.abs(error_noMotions)
    error_noMotions   = torch.mean(torch.mean(error_noMotions))
    
    return error_predictions, error_noMotions

# Function to extract all losses and errors for KVAE (KVAE_D will have an 
# additional d_loss)
# INPUTS:
# - a_mu: latent state 'a' of vae, mean
# - a_var: latent state 'a' of vae, covariance
# - reconstructedImagesBatch: VAE reconstructed images of the batch
# - currentImagesBatch: real images of the bacth
#   finding cross-entropy loss. IGNORE THIS. It is for another code version.
# - smooth: tuple of mean and covariance of latent state 'z'. 
# - A, B, C : matrices of KVAE,
# - kvae: kvae object
def ExtractLossesAndErrors(a_mu, a_var, reconstructedImagesBatch, currentImagesBatch, smooth, A, B, C, kvae, 
                           mu_filts, Sigma_filts, alpha_plot):
    
    # VAE loss (reconstruction + KLD)
    KLDLoss, ReconstructionLoss = VAE.VAE.FindVAELosses(a_mu, a_var, reconstructedImagesBatch, currentImagesBatch)
        
    mu_smooth, sigma_smooth = smooth
        
    #######################################################################
    # Find emission and transition loss    
    
    backward_states = smooth
    
    # with bhattacharya distance
    transition_loss_bhattacharya, predicted_values_bhattacharya, correct_values_bhattacharya = \
       kvae.CalculateBhattacharyaTransitionLoss(backward_states, A, B)
    emission_loss_bhattacharya                                                               = \
       kvae.CalculateBhattacharyaEmissionLoss(backward_states, C)
       
    smoothingFilteringEqualityLoss = kvae.CalculateSmoothingFilteringEqualityLoss(backward_states, mu_filts, Sigma_filts, A, B)
    
    tripletLossZ = kvae.CalculateTripletLossZ(backward_states, alpha_plot)
            
    #######################################################################
    # Which do we chose to train?
    predicted_values = predicted_values_bhattacharya
    correct_values   = correct_values_bhattacharya

    #######################################################################
    # Compare prediction against no prediction
    # Compare that predicting with matrix A is better than just picking the previous value
    error_predictions, error_noMotions = ComparePredictionsVsNoMotion(mu_smooth, predicted_values, 
                                                                      correct_values, kvae)
    
    return KLDLoss, ReconstructionLoss, transition_loss_bhattacharya, emission_loss_bhattacharya, \
           error_predictions, error_noMotions, smoothingFilteringEqualityLoss, tripletLossZ
           
# Function to save all the losses and the variables we are interested in, inside
# the summary holder, for KVAE case.
def SaveKVAEParamsAndLossesToSummaryHolder(ReconstructionLoss, KLDLoss, emission_loss, transition_loss, 
                                           error_predictions, error_noMotions, smoothingFilteringEqualityLoss, tripletLossZ, A, B, C, 
                                           a_mu, mu_smooth, alpha_plot, currentOdometryBatch, summaryCurrentEpoch, 
                                           mean_of_alpha_max):
    
    summaryCurrentEpoch.AppendValueInSummary('Reconstruction_losses', ReconstructionLoss.cpu().detach().numpy())  
    summaryCurrentEpoch.AppendValueInSummary('KLD_losses', KLDLoss.cpu().detach().numpy())  

    summaryCurrentEpoch.AppendValueInSummary('Emission_losses', emission_loss.cpu().detach().numpy())  
    summaryCurrentEpoch.AppendValueInSummary('Transition_losses', transition_loss.cpu().detach().numpy())  
    summaryCurrentEpoch.AppendValueInSummary('Prediction_error', error_predictions.cpu().detach().numpy())  
    summaryCurrentEpoch.AppendValueInSummary('No_motion_error', error_noMotions.cpu().detach().numpy()) 
    summaryCurrentEpoch.AppendValueInSummary('smoothingFilteringEqualityLoss', smoothingFilteringEqualityLoss.cpu().detach().numpy()) 
    summaryCurrentEpoch.AppendValueInSummary('tripletLossZ', tripletLossZ.cpu().detach().numpy()) 
    
    summaryCurrentEpoch.AppendValueInSummary('As', A.cpu().detach().numpy())  
    summaryCurrentEpoch.AppendValueInSummary('Bs', B.cpu().detach().numpy())  
    summaryCurrentEpoch.AppendValueInSummary('Cs', C.cpu().detach().numpy())  
    summaryCurrentEpoch.AppendValueInSummary('a_states', a_mu.cpu().detach().numpy())  
    summaryCurrentEpoch.AppendValueInSummary('z_states', mu_smooth.cpu().detach().numpy()) 
    
    summaryCurrentEpoch.AppendValueInSummary('alphas', alpha_plot.cpu().detach().numpy()) 
    summaryCurrentEpoch.AppendValueInSummary('odometry', currentOdometryBatch.cpu().detach().numpy())
    
    summaryCurrentEpoch.AppendValueInSummary('mean_of_alpha_max', mean_of_alpha_max.cpu().detach().numpy())
    
    return summaryCurrentEpoch

# Function to save all the losses and the variables we are interested in, inside
# the summary holder, for KVAE_D case.
def SaveKVAEDParamsAndLossesToSummaryHolder(ReconstructionLoss, KLDLoss, emission_loss, transition_loss, 
                                           error_predictions, error_noMotions, smoothingFilteringEqualityLoss, tripletLossZ, A, B, C, 
                                           a_mu, mu_smooth,  alpha_plot, currentOdometryBatch, summaryCurrentEpoch,
                                           mean_of_alpha_max, 
                                           d_loss, d_loss_denorm, d_loss_denorm_winning, 
                                           currentParamsBatch, predicted_params, predicted_params_min_error):
    
    summaryCurrentEpoch = SaveKVAEParamsAndLossesToSummaryHolder(ReconstructionLoss, KLDLoss, emission_loss, transition_loss, 
                                               error_predictions, error_noMotions, smoothingFilteringEqualityLoss, tripletLossZ, A, B, C, 
                                               a_mu, mu_smooth, alpha_plot, currentOdometryBatch, summaryCurrentEpoch, mean_of_alpha_max)
    
    summaryCurrentEpoch.AppendValueInSummary('d_losses', d_loss.cpu().detach().numpy()) 
    summaryCurrentEpoch.AppendValueInSummary('d_losses_denorm', d_loss_denorm.cpu().detach().numpy()) 
    summaryCurrentEpoch.AppendValueInSummary('d_loss_denorm_winning', d_loss_denorm_winning.cpu().detach().numpy()) 
    summaryCurrentEpoch.AppendValueInSummary('real_params', currentParamsBatch.cpu().detach().numpy()) 
    summaryCurrentEpoch.AppendValueInSummary('predicted_params_all', predicted_params.cpu().detach().numpy()) 
    summaryCurrentEpoch.AppendValueInSummary('predicted_params_min', predicted_params_min_error.cpu().detach().numpy())
    
    return summaryCurrentEpoch

# Function to find the mean value of the highest probability of clusters
# in a batch.
def FindMeanOfAlphaMaxs(alpha_plot):
    
    # Resize the alpha so that the batch size and the sequence length are 
    # together.
    # (batch size, sequence length, number of clusters)
    # -> 
    # (batch size*sequence length, number of clusters)
    alpha_reshaped = torch.reshape(alpha_plot, 
                                  (alpha_plot.shape[0]*alpha_plot.shape[1], alpha_plot.shape[2]))
    # Find max for each datapoint
    alpha_maxs = torch.max(alpha_reshaped, dim = 1).values
    # Find mean of max values
    mean_of_alpha_max = torch.mean(alpha_maxs)
    
    return mean_of_alpha_max

###############################################################################
# Functions to extract the mean and covariance over z at the end of an epoch
    
def CalculateMeanInAClusterGivenAllEpochStates(z_states, clusterAssignments, clusterIndex):
    
    # Extract values of the cluster
    currentClusterValues = z_states[clusterAssignments==clusterIndex]                        
    # Find mean of cluster values
    currentClusterMean = np.mean(currentClusterValues, axis = 0)
    
    return currentClusterMean, currentClusterValues

def CalculateCovInAClusterGivenAllEpochStates(z_states, clusterAssignments, clusterIndex):
    
    # Extract values of the cluster
    currentClusterValues = z_states[clusterAssignments==clusterIndex]                        
    # Find mean of cluster values
    currentClusterCov = np.cov(currentClusterValues)
    
    return currentClusterCov, currentClusterValues
    
def ExtractMeansAndCovariancesOfZStatesOverEpoch(summaryCurrentEpoch, config, inverseIndexing):
    
    # Find mean of z and covariance of z for each cluster, including the derivative information
    # 1) Extract the z states in the entire epoch
    z_states = summaryCurrentEpoch.FlattenValuesGivenKey(key = 'z_states', dataStructure = 0, batchSize = config['batch_size'], 
                                                                 inverseIndexing = inverseIndexing)
    # 2) Extract the probabilities of each cluster (alphas vector)
    alphas = summaryCurrentEpoch.FlattenValuesGivenKey(key = 'alphas', dataStructure = 0, batchSize = config['batch_size'], 
                                                                 inverseIndexing = inverseIndexing)
    # From the alphas obtain the cluster of belonging
    clusterAssignments = d_utils.FindHighestValuesAlongDimension(alphas, 1)
    # From the z_states obtain the velocity
    zero_array = np.zeros((1,z_states.shape[1]))
    z_states_vel = np.diff(z_states, axis = 0)
    z_states_vel = np.concatenate((zero_array, z_states_vel))

    # Now find mean and covariance for each cluster
    meanPerCluster = np.zeros((alphas.shape[1],z_states.shape[1]))
    meanVelPerCluster = np.zeros((alphas.shape[1],z_states.shape[1]))
    covPerCluster = []
    covVelPerCluster = []
    
    for i in range(alphas.shape[1]):
        
        # Find mean of cluster values
        currentClusterMean, _ = CalculateMeanInAClusterGivenAllEpochStates(z_states, clusterAssignments, i)
        currentClusterVelMean, _ = CalculateMeanInAClusterGivenAllEpochStates(z_states_vel, clusterAssignments, i)
        # Find covariances
        currentClusterCov, _ = CalculateCovInAClusterGivenAllEpochStates(z_states, clusterAssignments, i)
        currentClusterVelCov, _ = CalculateCovInAClusterGivenAllEpochStates(z_states_vel, clusterAssignments, i)
        
        meanPerCluster[i,:] = currentClusterMean
        meanVelPerCluster[i,:] = currentClusterVelMean
        covPerCluster.append(currentClusterCov)
        covVelPerCluster.append(currentClusterVelCov)
        
    return meanPerCluster, meanVelPerCluster, covPerCluster, covVelPerCluster, z_states, z_states_vel, clusterAssignments

# Function for plotting the transition matrix graph together with the mean 
# velocity of the clusters (over a subset of z)
def PlottingSubsetOfZState(meanPerCluster, meanVelPerCluster, transitionMatrix, file):
    
    fig = figure.Figure()
    ax = fig.subplots(1)
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    
    # z centers
    ax.scatter(meanPerCluster[:, 0], meanPerCluster[:, 1], color = 'red')
    # z velocity means
    ax.quiver(meanPerCluster[:, 0], meanPerCluster[:, 1], meanVelPerCluster[:, 0], meanVelPerCluster[:, 1], color = 'black')
    # Connect with a quiver the means of clusters that are connected
    nonZeroElementsTransitionMat = torch.nonzero(transitionMatrix)
    connection_cluster_x = nonZeroElementsTransitionMat[:,0].detach().cpu().numpy()
    connection_cluster_y = nonZeroElementsTransitionMat[:,1].detach().cpu().numpy()
    ax.quiver(meanPerCluster[connection_cluster_x, 0] , meanPerCluster[connection_cluster_x, 1], 
              meanPerCluster[connection_cluster_y, 0] - meanPerCluster[connection_cluster_x, 0], 
              meanPerCluster[connection_cluster_y, 1] - meanPerCluster[connection_cluster_x, 1], 
              color = 'blue', scale_units='xy', angles='xy', scale=1)    
    fig.savefig(file)    
    return
            
###############################################################################

# Function for training the KVAE OVER A RANGE OF EPOCHS.
# INPUTS:
# - kvae: initialized kvae;
# - config: configuration dictionary,
# - trainingData: training data, in modified sequential order, 
# - mask_train: mask of training (in our code it is an array of all 1s)
# - numberOfDataBatches: number of data batches
# - beginEpoch: epoch where we begin the training in the function (e.g., if we are 
#   loading an already half-trained model and we want to continue the training
#   from there).
# - inverseIndexing: indices to go back to the original order of the data sequences.
# OUTPUTS:
# - kvae: trained kvae 
# - summaryTrainingAllEpochs: summary holder for all training epochs
# - summaryCurrentEpoch: summary holder for current epoch
def TrainingLoopKVAE(kvae, config, trainingData, mask_train, numberOfDataBatches, 
                     beginEpoch, inverseIndexing):
    
    # Initial learning rate
    learningRate = config['init_lr']
    learningRate_VAE = config['init_lr_VAE']
    
    # Summary of training, for all epochs
    summaryTrainingAllEpochs    = SHLAE.SummaryHolderLossesAcrossEpochs(summaryNamesAllEpochs)
    
    # ------------------------- TRAINING THE KVAE -----------------------------    
    # Looping over the number of epochs
    for n in range(config['only_vae_epochs'],config['train_epochs']):
          
        print('Epoch: ' + str(n))
        
        kvae.train()
        
        # Set this as first time in KF
        kvae.kf.setFirstTimeInstantOfSequence()
        
        # Summary of training losses and values, for current epoch
        summaryCurrentEpoch = SH.SummaryHolder(summaryNamesCurrentEpoch)
        
        # Modify learning rate
        if n > 0:
            globalStep   = n
            learningRate = learningRate*np.power(config['decay_rate'], globalStep/config['decay_steps'])
            learningRate_VAE = learningRate_VAE*np.power(config['decay_rate'], globalStep/config['decay_steps'])
        # Put the current learning rate in the summary
        summaryCurrentEpoch.AppendValueInSummary('learning_rates', learningRate)  
        
        # Increment the skewness value
        kvae.IncrementSkewness(config['skew_increase_per_epoch'])
            
        # Random image to print in this epoch
        randomBatchToPrint = np.random.randint(numberOfDataBatches)
        
        kvae.baseVAE.FreezeBatchNormLayers()
    
        # Looping over the batches of training data
        for i in range(numberOfDataBatches):
            
            print('Batch number ' + str(i+1) + ' out of ' + str(numberOfDataBatches))
            
            # Prepare the input data
            currentImagesBatch, currentControlsBatch, currentOdometryBatch, \
               currentParamsBatch, currentDistancesBatch, mask = \
               kvae.PrepareInputsForKVAETraining(trainingData=trainingData, batchSize = config['batch_size'], 
                                                 currentBatchNumber=i, mask_train=mask_train, 
                                                 image_channels = config['image_channels'])
            
            # Change the learning rate of optimizer
            if n >= config['only_vae_epochs'] and n < config['only_vae_epochs'] + config['kf_update_steps']:
                list_of_params = kvae.kf.parameters()
                optimizer      = torch.optim.Adam(list_of_params, lr =learningRate, weight_decay = config['weight_decay'])
            else:
                optimizer      = torch.optim.SGD([{'params' : kvae.kf.parameters()},
                                             {'params' : kvae.baseVAE.parameters(), 'lr': learningRate_VAE}],
                                             lr =learningRate, weight_decay = config['weight_decay'])                 
            ######################### CALL TO KVAE ################################ 
            # forward step
            reconstructedImagesBatch, a_seq, a_mu, a_var, smooth, A, B, C, alpha_plot, mu_preds, Sigma_preds, mu_filts, Sigma_filts = \
               kvae.PerformKVAESmoothingOverBatchSequence(currentImagesBatch,currentDistancesBatch,mask)
            mu_smooth, sigma_smooth = smooth
            ########################## FIND LOSSES ################################ 
            KLDLoss, ReconstructionLoss, transition_loss, emission_loss, error_predictions, error_noMotions, smoothingFilteringEqualityLoss, tripletLossZ = \
                   ExtractLossesAndErrors(a_mu, a_var, reconstructedImagesBatch, currentImagesBatch, smooth, A, B, C, kvae, 
                                          mu_filts, Sigma_filts, alpha_plot)
            ################## FIND MEAN of Highest alpha #########################
            mean_of_alpha_max = FindMeanOfAlphaMaxs(alpha_plot)
            #######################################################################
            # Printing image reconstruction temporary results
            if i == randomBatchToPrint:
                VAE.VAE.PrintRealVsReconstructedImages(currentImagesBatch[0,:,0,:,:], reconstructedImagesBatch[0,:,0,:,:], 
                                                       config['output_folder'] + '/TRAIN_IMGS_' + str(n) + '.png')  
            #######################################################################
            # OPTIMIZER            
            optimizer.zero_grad()            
            if n >= config['only_vae_epochs'] and n < config['only_vae_epochs'] + config['kf_update_steps']: # Remember <= or it will start with the third phase and ruin the training!!
                print('Optimize KF only')
                kf_loss  = config['transitionLoss_weight']*transition_loss + config['emissionLoss_weight']*emission_loss + \
                           config['transitionLoss_weight']*smoothingFilteringEqualityLoss #+ config['transitionLoss_weight']*tripletLossZ/500
                kf_loss.backward()
                loss_tot = kf_loss
            else:
                loss_tot = config['KLDLoss_weight']*KLDLoss + config['RecLoss_weight']*ReconstructionLoss + \
                           config['transitionLoss_weight']*transition_loss + config['emissionLoss_weight']*emission_loss + \
                           config['transitionLoss_weight']*smoothingFilteringEqualityLoss #+ config['transitionLoss_weight']*tripletLossZ/500
                print('Optimize KVAE')    
                loss_tot.backward()  
                kf_loss = loss_tot
                
            torch.nn.utils.clip_grad_norm_(kvae.baseVAE.parameters(), config['max_grad_norm_VAE'])
            torch.nn.utils.clip_grad_norm_(kvae.kf.parameters(), config['max_grad_norm_kf'])
            optimizer.step()            
            #######################################################################
            # Inserting all losses and other variables we want to save in the 
            # summary holders.
            summaryCurrentEpoch = SaveKVAEParamsAndLossesToSummaryHolder(ReconstructionLoss, KLDLoss, emission_loss, transition_loss, 
                                                       error_predictions, error_noMotions, smoothingFilteringEqualityLoss, tripletLossZ, A, B, C, 
                                                       a_mu, mu_smooth, alpha_plot, currentOdometryBatch, summaryCurrentEpoch, 
                                                       mean_of_alpha_max)            
            #######################################################################
            del reconstructedImagesBatch
            del KLDLoss, ReconstructionLoss
            del kf_loss, loss_tot
            del mu_smooth, sigma_smooth
            del a_seq, a_mu, a_var, alpha_plot 
            del smooth, A ,B ,C 
            del transition_loss, emission_loss
            del smoothingFilteringEqualityLoss
            del error_predictions, error_noMotions
            del currentImagesBatch
            del currentOdometryBatch
            del currentDistancesBatch        
            del currentParamsBatch
            del currentControlsBatch        
            del optimizer
            
            if device.type == "cuda":
                torch.cuda.empty_cache() 
            # End of batch loop
                
        ###########################################################################
        ###########################################################################
        # Handle the losses over TRAINING epochs
        # Add the mean of the losses of the current epoch to the overall summary
        summaryTrainingAllEpochs.AppendToOverallSummaryMeanValuesOfCurrentEpochSummary(summaryCurrentEpoch)
        # Plot losses
        summaryTrainingAllEpochs.PlotValuesInSummaryAcrossTime(outputFolder = config['output_folder'], filePrefix = 'TRAIN_PLOT_')
        # Save losses to matlab
        summaryTrainingAllEpochs.BringValuesToMatlab(outputFolder = config['output_folder'], filePrefix = 'TRAIN_')
        
        ###########################################################################
        # Save to matlab the values over the current epoch
        summaryCurrentEpoch.BringValuesToMatlab(outputFolder = config['output_folder'], dataStructure = 0, 
                                                batchSize = config['batch_size'], filePrefix = 'TRAIN_single_epoch_' + str(n) + '_' , 
                                                inverseIndexing = inverseIndexing)
        
        ###########################################################################          
        # Save the models
        torch.save(kvae.state_dict(), config['output_folder'] + '/kvae.torch')
        torch.save(kvae.state_dict(), config['output_folder'] + '/kvae_' + str(n) + '.torch')
        
    return kvae, summaryTrainingAllEpochs, summaryCurrentEpoch

# Function for training the KVAE_D OVER A SINGLE EPOCH.
# INPUTS:
# - kvae: initialized kvae;
# - config: configuration dictionary,
# - trainingData: training data, in modified sequential order, 
# - currentEpoch: current epoch of training
# - learningRate: learning rate of parameters of KF
# - learningRate_VAE: learning rate of parameters of VAE
# - numberOfDataBatches: number of data batches
# OUTPUTS:
# - kvae: trained kvae 
# - summaryCurrentEpoch: summary holder for current epoch
def TrainingOneEpochKVAE_D(kvae, config, trainingData, currentEpoch, learningRate, learningRate_VAE,
                           numberOfDataBatches, trainingDistances):
    
    kvae.train()        
    print('Epoch: ' + str(currentEpoch))    
    # Set this as first time in KF
    kvae.kf.setFirstTimeInstantOfSequence()    
    # Summary of training losses and values, for current epoch
    summaryCurrentEpoch = SH.SummaryHolder(summaryNamesCurrentEpoch)
    
    # Modify learning rate
    if currentEpoch > 0:
        globalStep   = currentEpoch
        learningRate = learningRate*np.power(config['decay_rate'], globalStep/config['decay_steps'])
        learningRate_VAE = learningRate_VAE*np.power(config['decay_rate'], globalStep/config['decay_steps'])
    # Put the current learning rate in the summary
    summaryCurrentEpoch.AppendValueInSummary('learning_rates', learningRate)  
        
    # Increment the skewness value 
    kvae.IncrementSkewness(config['skew_increase_per_epoch'])        
    # Random image to print in this epoch
    randomBatchToPrint = np.random.randint(numberOfDataBatches)
    
    # Looping over the batches of training data
    for i in range(numberOfDataBatches):
        
        print('Batch number ' + str(i+1) + ' out of ' + str(numberOfDataBatches))
        
        # Prepare the input data
        currentImagesBatch, currentControlsBatch, currentOdometryBatch, \
           currentParamsBatch, currentDistancesBatch = \
           kvae.PrepareInputsForKVAE(data=trainingData, distances = trainingDistances, batchSize = config['batch_size'], 
                                     currentBatchNumber = i, image_channels = config['image_channels'])
           
        # Change the learning rate of optimizer
        if currentEpoch >= config['only_vae_epochs'] and currentEpoch < config['only_vae_epochs'] + config['kf_update_steps']:
            list_of_params = kvae.kf.parameters()  
            optimizer      = torch.optim.SGD(list_of_params, lr = learningRate, weight_decay = config['weight_decay'])
        else:
            #optimizer      = torch.optim.SGD(kvae.parameters(), lr =learningRate, weight_decay = config['weight_decay'])
            optimizer      = torch.optim.SGD([{'params' : kvae.kf.parameters()},
                                             {'params' : kvae.baseVAE.parameters(), 'lr': learningRate_VAE}],
                                             lr =learningRate, weight_decay = config['weight_decay'])
            
        ######################### CALL TO KVAE ################################ <------------------------------------
        # forward step
        reconstructedImagesBatch, a_seq, a_mu, a_var, smooth, A, B, C, alpha_plot, params_pred, mu_preds, Sigma_preds, mu_filts, Sigma_filts = \
           kvae.PerformKVAESmoothingOverBatchSequence(currentImagesBatch,currentDistancesBatch)
        mu_smooth, sigma_smooth = smooth
        predicted_params = params_pred        
        ########################## FIND LOSSES ################################ 
        KLDLoss, ReconstructionLoss, transition_loss, emission_loss, error_predictions, error_noMotions, smoothingFilteringEqualityLoss, tripletLossZ = \
               ExtractLossesAndErrors(a_mu, a_var, reconstructedImagesBatch, currentImagesBatch, smooth, A, B, C, kvae, 
                                      mu_filts, Sigma_filts, alpha_plot)
        # Loss on prediction of odometry from video                     
        d_loss = KVAE_D.KalmanVariationalAutoencoder_D.CalculateDMatrixLoss(currentParamsBatch, params_pred)
        d_loss_denorm = KVAE_D.KalmanVariationalAutoencoder_D.CalculateDMatrixLossDenormalized(
                currentParamsBatch, params_pred, 
                maxXReal = config['maxXReal'], maxYReal = config['maxYReal'], 
                minXReal = config['minXReal'], minYReal = config['minYReal'])
        d_loss_denorm_winning = kvae.CalculateDMatrixLossBestClusterDenormalized(
                currentParamsBatch, mu_smooth, alpha_plot,
                maxXReal = config['maxXReal'], maxYReal = config['maxYReal'], 
                minXReal = config['minXReal'], minYReal = config['minYReal'])
        predicted_params_min_error = predicted_params       
        ################## FIND MEAN of Highest alpha #########################
        mean_of_alpha_max = FindMeanOfAlphaMaxs(alpha_plot)
        #######################################################################
        # Printing image reconstruction temporary results
        if i == randomBatchToPrint:            
           VAE.VAE.PrintRealVsReconstructedImages(currentImagesBatch[0,:,0,:,:], reconstructedImagesBatch[0,:,0,:,:], 
                                                       config['output_folder'] + '/TRAIN_IMGS_' + str(currentEpoch) + '.png')            
        #######################################################################
        # OPTIMIZER  
        optimizer.zero_grad()
        
        if currentEpoch >= config['only_vae_epochs'] and currentEpoch < config['only_vae_epochs'] + config['kf_update_steps']: # Remember <= or it will start with the third phase and ruin the training!!
            print('Optimize KF only')
            kf_loss  = config['transitionLoss_weight']*transition_loss + config['emissionLoss_weight']*emission_loss + \
                       config['dLoss_weight']*d_loss + config['transitionLoss_weight']*smoothingFilteringEqualityLoss #+ config['transitionLoss_weight']*tripletLossZ/500
            kf_loss.backward()
            loss_tot = kf_loss
        else:
            loss_tot = config['KLDLoss_weight']*KLDLoss + config['RecLoss_weight']*ReconstructionLoss + \
                       config['transitionLoss_weight']*transition_loss + config['emissionLoss_weight']*emission_loss + \
                       config['dLoss_weight']*d_loss + config['transitionLoss_weight']*smoothingFilteringEqualityLoss #+ config['transitionLoss_weight']*tripletLossZ/500
            print('Optimize KVAE') 
            loss_tot.backward()  
            kf_loss = loss_tot
            
        torch.nn.utils.clip_grad_norm_(kvae.baseVAE.parameters(), config['max_grad_norm_VAE'])
        torch.nn.utils.clip_grad_norm_(kvae.kf.parameters(), config['max_grad_norm_kf'])
        optimizer.step()
        #######################################################################
        summaryCurrentEpoch = SaveKVAEDParamsAndLossesToSummaryHolder(ReconstructionLoss, KLDLoss, emission_loss, transition_loss, 
                                                   error_predictions, error_noMotions, smoothingFilteringEqualityLoss, tripletLossZ,
                                                   A, B, C, a_mu, mu_smooth,  alpha_plot, 
                                                   currentOdometryBatch, summaryCurrentEpoch, mean_of_alpha_max, d_loss, d_loss_denorm, 
                                                   d_loss_denorm_winning, currentParamsBatch, predicted_params, predicted_params_min_error)
        #######################################################################
        del reconstructedImagesBatch
        del KLDLoss, ReconstructionLoss
        del kf_loss, loss_tot
        del mu_smooth, sigma_smooth
        del a_seq, a_mu, a_var, alpha_plot
        del smooth, A ,B ,C
        del predicted_params, predicted_params_min_error
        del d_loss, d_loss_denorm, d_loss_denorm_winning
        del mean_of_alpha_max
        del transition_loss, emission_loss
        del smoothingFilteringEqualityLoss
        del error_predictions, error_noMotions
        del currentImagesBatch
        del currentOdometryBatch
        del currentDistancesBatch        
        del currentParamsBatch
        del currentControlsBatch        
        del optimizer   
        
        if device.type == "cuda":
            torch.cuda.empty_cache()    
        # end of batch
    
    return kvae, summaryCurrentEpoch

# Function for training the KVAE_D OVER A RANGE OF EPOCHS.
# INPUTS:
# - kvae: initialized kvae;
# - config: configuration dictionary,
# - trainingData: training data, in modified sequential order, 
# - mask_train: mask of training (in our code it is an array of all 1s)
# - numberOfDataBatches: number of data batches
# - beginEpoch: epoch where we begin the training in the function (e.g., if we are 
#   loading an already half-trained model and we want to continue the training
#   from there).
# - inverseIndexing: indices to go back to the original order of the data sequences.
# OUTPUTS:
# - kvae: trained kvae 
# - summaryTrainingAllEpochs: summary holder for all training epochs
# - summaryCurrentEpoch: summary holder for current epoch
def TrainingLoopKVAE_D(kvae, config, trainingData, numberOfDataBatches, beginEpoch, inverseIndexing, trainingDistances):
    
    # Initial learning rate
    learningRate = config['init_lr']
    learningRate_VAE = config['init_lr_VAE']
    
    # Summary of training, for all epochs
    summaryTrainingAllEpochs    = SHLAE.SummaryHolderLossesAcrossEpochs(summaryNamesAllEpochs)
        
    # Looping over the number of epochs
    for n in range(beginEpoch,config['train_epochs']):
        
        kvae, summaryTrainingCurrentEpoch = TrainingOneEpochKVAE_D(kvae, config, trainingData, n, learningRate, 
                                                                   learningRate_VAE, numberOfDataBatches, 
                                                                   trainingDistances)
        # Handle the losses over TRAINING epochs
        summaryTrainingAllEpochs.PerformFinalBatchOperations(summaryTrainingCurrentEpoch, 
                                                           config['output_folder'], filePrefix = 'TRAIN_')        
        # Save to matlab the values over the current epoch
        summaryTrainingCurrentEpoch.BringValuesToMatlab(outputFolder = config['output_folder'], 
                                                dataStructure = 0, batchSize = config['batch_size'], 
                                                filePrefix = 'TRAIN_single_epoch_', 
                                                inverseIndexing = inverseIndexing)      
        # Save the models
        torch.save(kvae.state_dict(), config['output_folder'] + '/kvae.torch')
        torch.save(kvae.state_dict(), config['output_folder'] + '/kvae_' + str(n) + '.torch')
        
        ###########################################################################  
        # PLOT of predicted params (odometry) vs. real one        
        # Real params
        real_params_for_printing = RD.reshape1DdataWithInverseIndexingAndNoTrajectoryDimension(
                summaryTrainingCurrentEpoch.summary['real_params'], inverseIndexing)
        # Predicted params
        predicted_params_min_for_printing = RD.reshape1DdataWithInverseIndexingAndNoTrajectoryDimension(
                summaryTrainingCurrentEpoch.summary['predicted_params_min'], inverseIndexing)        
        # SCATTERPLOT
        PG.plot_predicted_vs_real_states(predicted_params_min_for_printing[:,0:2], real_params_for_printing[:,0:2],
                                         config['output_folder'] + '/TRAIN_params_pred_trajectory_' + str(n) + '.png')
        # DELETES
        if n != config['train_epochs'] - 1:
            del summaryTrainingCurrentEpoch
        del real_params_for_printing, predicted_params_min_for_printing
        if device.type == "cuda":
            torch.cuda.empty_cache()      
        
    return kvae, summaryTrainingAllEpochs, summaryTrainingCurrentEpoch


# Function for testing the KVAE_D OVER A SINGLE EPOCH.
# INPUTS:
# - kvae: initialized kvae;
# - config: configuration dictionary,
# - testingData: testing data, in modified sequential order, 
# - currentEpoch: epoch of current testing,
# - mask_test: mask of testing (in our code it is an array of all 1s)
# - numberOfDataBatches: number of data batches
# OUTPUTS:
# - summaryCurrentEpoch: summary holder for current epoch
def TestingOneEpochKVAE_D(kvae, config, testingData, currentEpoch, numberOfDataBatches, testingDistances):
    
    with torch.no_grad():
        
        kvae.eval()        
        print('Epoch TEST: ' + str(currentEpoch))        
        # Set this as first time in KF
        kvae.kf.setFirstTimeInstantOfSequence()        
        # Summary of training losses and values, for current epoch
        summaryCurrentEpoch = SH.SummaryHolder(summaryNamesCurrentEpoch)            
        # Random image to print in this epoch
        randomBatchToPrint = np.random.randint(numberOfDataBatches)
        
        # Looping over the batches of training data
        for i in range(numberOfDataBatches):
            
            print('Batch number ' + str(i+1) + ' out of ' + str(numberOfDataBatches))
            
            # Prepare the input data
            currentImagesBatch, currentControlsBatch, currentOdometryBatch, \
               currentParamsBatch, currentDistancesBatch = \
               kvae.PrepareInputsForKVAE(data = testingData, distances = testingDistances, batchSize = config['batch_size'], 
                                         currentBatchNumber = i, image_channels = config['image_channels'])              
            ######################### CALL TO KVAE ################################ <------------------------------------           
            # forward step
            reconstructedImagesBatch, a_seq, a_mu, a_var, smooth, A, B, C, alpha_plot, params_pred, \
               mu_preds, Sigma_preds, mu_filts, Sigma_filts = \
                  kvae.PerformKVAESmoothingOverBatchSequence(currentImagesBatch,currentDistancesBatch)               
            mu_smooth, sigma_smooth = smooth
            predicted_params = params_pred
            predicted_params_min_error = predicted_params            
            ########################## FIND LOSSES ################################ 
            KLDLoss, ReconstructionLoss, transition_loss, emission_loss, error_predictions, error_noMotions, smoothingFilteringEqualityLoss, tripletLossZ = \
                   ExtractLossesAndErrors(a_mu, a_var, reconstructedImagesBatch, currentImagesBatch, smooth, A, B, C, kvae, 
                                          mu_filts, Sigma_filts,alpha_plot)
            # Loss on prediction of odometry from video                     
            d_loss = KVAE_D.KalmanVariationalAutoencoder_D.CalculateDMatrixLoss(currentParamsBatch, params_pred)   
            d_loss_denorm = KVAE_D.KalmanVariationalAutoencoder_D.CalculateDMatrixLossDenormalized(
                currentParamsBatch, params_pred, 
                maxXReal = config['maxXReal'], maxYReal = config['maxYReal'], 
                minXReal = config['minXReal'], minYReal = config['minYReal'])
            d_loss_denorm_winning = kvae.CalculateDMatrixLossBestClusterDenormalized(
                currentParamsBatch, mu_smooth, alpha_plot,
                maxXReal = config['maxXReal'], maxYReal = config['maxYReal'], 
                minXReal = config['minXReal'], minYReal = config['minYReal'])
            ################## FIND MEAN of Highest alpha #########################
            mean_of_alpha_max = FindMeanOfAlphaMaxs(alpha_plot)
            #######################################################################
            # Printing image reconstruction temporary results
            if i == randomBatchToPrint:
                VAE.VAE.PrintRealVsReconstructedImages(currentImagesBatch[0,:,0,:,:], reconstructedImagesBatch[0,:,0,:,:], 
                                                       config['output_folder'] + '/TRAIN_IMGS_' + str(currentEpoch) + '.png')            
            #######################################################################
            summaryCurrentEpoch = SaveKVAEDParamsAndLossesToSummaryHolder(ReconstructionLoss, KLDLoss, emission_loss, transition_loss, 
                                                       error_predictions, error_noMotions, smoothingFilteringEqualityLoss, tripletLossZ,
                                                       A, B, C, a_mu, mu_smooth,  alpha_plot, 
                                                       currentOdometryBatch, summaryCurrentEpoch, mean_of_alpha_max, 
                                                       d_loss, d_loss_denorm, d_loss_denorm_winning, 
                                                       currentParamsBatch, predicted_params, predicted_params_min_error)           
            #######################################################################
            del reconstructedImagesBatch
            del KLDLoss, ReconstructionLoss
            del mu_smooth, sigma_smooth
            del a_seq, a_mu, a_var, alpha_plot, params_pred
            del smooth, A ,B ,C 
            del predicted_params, predicted_params_min_error
            del d_loss, d_loss_denorm, d_loss_denorm_winning
            del mean_of_alpha_max
            del transition_loss, emission_loss
            del smoothingFilteringEqualityLoss
            del error_predictions, error_noMotions
            del currentImagesBatch
            del currentOdometryBatch
            del currentDistancesBatch        
            del currentParamsBatch
            del currentControlsBatch                 
            if device.type == "cuda":
                torch.cuda.empty_cache()      
            # end of batch
    
    return summaryCurrentEpoch
        
# Loop of both training and testing of KVAE_D.
# INPUTS:
# - kvae: initialized kvae;
# - config: configuration dictionary,
# - trainingData: training data, in modified sequential order, 
# - testingData: testing data, in modified sequential order, 
# - mask_train: mask of training (in our code it is an array of all 1s)
# - mask_test: mask of testing (in our code it is an array of all 1s)
# - numberOfDataBatches: number of data batches
# - numberOfDataBatchesTest: number of data batches for testing
# - beginEpoch: epoch where we begin the training in the function (e.g., if we are 
#   loading an already half-trained model and we want to continue the training
#   from there).
# - inverseIndexing: indices to go back to the original order of the data sequences, 
#   for training data.
# - inverseIndexingTest: indices to go back to the original order of the data sequences,
#   for testing data.
# OUTPUTS:
# - kvae: trained kvae 
# - summaryTrainingAllEpochs: summary holder for all training epochs
# - summaryCurrentEpoch: summary holder for current epoch
def TrainingAndValidationLoopKVAE_D(kvae, config, trainingData, testingData,
                                 numberOfDataBatches, numberOfDataBatchesTest, beginEpoch, 
                                 inverseIndexing, inverseIndexingTest, 
                                 trainingDistances, testingDistances):

    # Initial learning rate
    learningRate = config['init_lr']
    learningRate_VAE = config['init_lr_VAE']    
    # Summary of training, for all epochs
    summaryTrainingAllEpochs    = SHLAE.SummaryHolderLossesAcrossEpochs(summaryNamesAllEpochs)
    summaryTestingAllEpochs     = SHLAE.SummaryHolderLossesAcrossEpochs(summaryNamesAllEpochs)
        
    # Looping over the number of epochs
    for n in range(beginEpoch,config['train_epochs']):
        ####################################################################### 
        print('TRAINING') 
        ####################################################################### 
        kvae, summaryTrainingCurrentEpoch = TrainingOneEpochKVAE_D(kvae, config, trainingData, 
                                                           n, learningRate, learningRate_VAE,
                                                           numberOfDataBatches, trainingDistances)   
        print('Updating summaries across train')
        
        # Handle the losses over TRAINING epochs
        summaryTrainingAllEpochs.PerformFinalBatchOperations(summaryTrainingCurrentEpoch, 
                                                           config['output_folder'], filePrefix = 'TRAIN_')        
        # Save to matlab the values over the current epoch
        summaryTrainingCurrentEpoch.BringValuesToMatlab(outputFolder = config['output_folder'], 
                                                        dataStructure = 0, 
                                                        batchSize = config['batch_size'], 
                                                        filePrefix = 'TRAIN_single_epoch_', 
                                                        inverseIndexing = inverseIndexing) 
        
        # Extract mean and covariance over z in the current epoch
        meanPerCluster, meanVelPerCluster, covPerCluster, covVelPerCluster, z_states, z_states_vel, clusterAssignments = \
            ExtractMeansAndCovariancesOfZStatesOverEpoch(summaryTrainingCurrentEpoch, config, inverseIndexing)
        # Plot
        PlottingSubsetOfZState(meanPerCluster, meanVelPerCluster, kvae.clusterGraph.transitionMat,
                               config['output_folder'] + '/Z_space_' + str(n) + '.png')
        
        #PG.PlottingZVelocitiesAndTheirMeans(meanVelPerCluster, z_states_vel, clusterAssignments, 
                                            #config['output_folder'] + '/Z_vel_space_' + str(n) + '.png')
        
        # Save the models
        print('Saving the model')
        torch.save(kvae.state_dict(), config['output_folder'] + '/kvae.torch')
        torch.save(kvae.state_dict(), config['output_folder'] + '/kvae_' + str(n) + '.torch')

        ###########################################################################  
        print('Plotting results of train in this epoch') 
        # PLOT of predicted params (odometry) vs. real one        
        # Real params        
        real_params_for_printing = RD.reshape1DdataWithInverseIndexingAndNoTrajectoryDimension(
                summaryTrainingCurrentEpoch.summary['real_params'], inverseIndexing)
        # Predicted params
        predicted_params_min_for_printing = RD.reshape1DdataWithInverseIndexingAndNoTrajectoryDimension(
                summaryTrainingCurrentEpoch.summary['predicted_params_min'], inverseIndexing)         
        # Scatterplot with quiver
        #PG.plot_predicted_vs_real_states_onScatterPlotWithQuivers(predicted_params_min_for_printing[:,0:2],
        #                                 real_params_for_printing[:,0:2],
        #                                 config['output_folder'] + '/TRAIN_params_pred_scatter_' + str(n) + '.png')        
        # SCATTERPLOT sequential
        fileName = config['output_folder'] + '/TRAIN_params_pred_trajectory_' + str(n) + '.png'
        PG.plot_predicted_vs_real_states(predicted_params_min_for_printing,
                                         real_params_for_printing, fileName)     
        # DELETES
        if n != config['train_epochs'] - 1:
            del summaryTrainingCurrentEpoch
        del real_params_for_printing, predicted_params_min_for_printing, fileName
        if device.type == "cuda":
            torch.cuda.empty_cache()  
        #######################################################################    
        print('VALIDATION')
        
        ####################################################################### 
        with torch.no_grad():
            summaryTestingCurrentEpoch = TestingOneEpochKVAE_D(kvae, config, testingData, n, 
                               numberOfDataBatchesTest, testingDistances)     
            print('Updating summaries across validation')
            # Handle the losses over TESTING epochs
            summaryTestingAllEpochs.PerformFinalBatchOperations(summaryTestingCurrentEpoch, 
                                                               config['output_folder'], filePrefix = 'VAL_')
            # Save to matlab the values over the current epoch
            summaryTestingCurrentEpoch.BringValuesToMatlab(outputFolder = config['output_folder'], 
                                                           dataStructure = 0, batchSize = config['batch_size_test'], 
                                                           filePrefix = 'VAL_single_epoch_', 
                                                           inverseIndexing = inverseIndexingTest)    
            ###########################################################################  
            print('Plotting results of validation in this epoch') 
            # PLOT of predicted params (odometry) vs. real one        
            # Real params
            real_params_for_printing = RD.reshape1DdataWithInverseIndexingAndNoTrajectoryDimension(
                    summaryTestingCurrentEpoch.summary['real_params'], inverseIndexingTest)
            # Predicted params
            predicted_params_min_for_printing = RD.reshape1DdataWithInverseIndexingAndNoTrajectoryDimension(
                    summaryTestingCurrentEpoch.summary['predicted_params_min'], inverseIndexingTest)        
            # SCATTERPLOT
            PG.plot_predicted_vs_real_states(predicted_params_min_for_printing, real_params_for_printing,
                                             config['output_folder'] + '/VAL_params_pred_trajectory_' + str(n) + '.png')
            # DELETES
            if n != config['train_epochs'] - 1:
                del summaryTestingCurrentEpoch
            del real_params_for_printing, predicted_params_min_for_printing
            if device.type == "cuda":
                torch.cuda.empty_cache()      
        
    return kvae, summaryTrainingAllEpochs, summaryTrainingCurrentEpoch, \
           summaryTestingAllEpochs, summaryTestingCurrentEpoch            
           
###############################################################################
# TRAINING OVERALL CODES
         
# Function to perform KVAE training with configuration file already loaded.
# INPUTS:
# - config: loaded configuration dictionary.
# OUTPUTS:
# - kvae: trained KVAE model object.
# - summaryTrainingAllEpochs: SummaryHolderLossesAcrossEpochs object over all epochs.
# - summaryTrainingCurrentEpoch: SummaryHolder object over last epoch.
# - summaryTestingAllEpochs: SummaryHolderLossesAcrossEpochs object over all 
#         epochs of testing.
# - summaryTestingCurrentEpoch: SummaryHolder object over last epoch of testing.
def TrainWithValidationGivenLoadedConfiguration(config):
    
    # Prepare training
    shuffledTrainingData, newIndicesShuffled, shuffledTestingData, newIndicesShuffledTesting, \
       trainingData, newIndices, inverseIndexing, testingData, newIndicesTesting, \
       inverseIndexingTest, clusterGraph, numberOfDataBatches, numberOfDataBatchesTest, \
       trainingDistances, testingDistances, kvae = PrepareEverythingForTrainingAndValidation(config)
    
    # Perform traning and testing of the VAE
    kvae.baseVAE, summaryTrainingAllEpochs, summaryTestingAllEpochs,summaryTrainingCurrentEpoch, summaryTestingCurrentEpoch = \
       PerformVAETrainingAndTesting(kvae.baseVAE, config, shuffledTrainingData, shuffledTestingData)  
    # Load KVAE with trained KF
    kvae, beginEpoch = LoadKVAEwithTrainedKF(kvae, config)

    # Perform training and testing loop of KVAE_D
    kvae, summaryTrainingAllEpochs, summaryTrainingCurrentEpoch, \
       summaryTestingAllEpochs, summaryTestingCurrentEpoch = \
       TrainingAndValidationLoopKVAE_D(kvae, config, trainingData, testingData, 
                                    numberOfDataBatches, numberOfDataBatchesTest, beginEpoch, 
                                    inverseIndexing, inverseIndexingTest, 
                                    trainingDistances, testingDistances)  

    return kvae, summaryTrainingAllEpochs, summaryTrainingCurrentEpoch, \
           summaryTestingAllEpochs, summaryTestingCurrentEpoch
           
# Function to perform KVAE training with configuration file already loaded.
# INPUTS:
# - config: loaded configuration dictionary.
# OUTPUTS:
# - kvae: trained KVAE model object.
# - summaryTrainingAllEpochs: SummaryHolderLossesAcrossEpochs object over all epochs.
# - summaryTrainingCurrentEpoch: SummaryHolder object over last epoch.
def trainGivenLoadedConfiguration(config):
    
    # Prepare training
    shuffledTrainingData, newIndicesShuffled, shuffledTestingData, newIndicesShuffledTesting, \
       trainingData, newIndices, inverseIndexing, testingData, newIndicesTesting, \
       inverseIndexingTesting, clusterGraph, numberOfDataBatches, numberOfDataBatchesTest, \
       trainingDistances, testingDistances, kvae = PrepareEverythingForTrainingAndValidation(config)
       
    # Perform traning and testing of the VAE
    kvae.baseVAE, summaryTrainingAllEpochs, summaryTestingAllEpochs, summaryTrainingCurrentEpoch, summaryTestingCurrentEpoch = \
       PerformVAETrainingAndTesting(kvae.baseVAE, config, shuffledTrainingData, shuffledTestingData) 
       
    # Load KVAE with trained KF
    kvae, beginEpoch = LoadKVAEwithTrainedKF(kvae, config)
    
    # Perform training loop of KVAE_D
    kvae, summaryTrainingAllEpochs, summaryTrainingCurrentEpoch = \
       TrainingLoopKVAE_D(kvae, config, trainingData, numberOfDataBatches, 
                          beginEpoch, inverseIndexing, trainingDistances)
    
    return kvae, summaryTrainingAllEpochs, summaryTrainingCurrentEpoch

# Function to perform both training and validation epoch by epoch.
# INPUTS:
# - loadedConfiguration: path to the configuration file to load.
# OUTPUTS:
# - kvae: trained KVAE model object.
# - summaryTrainingAllEpochs: SummaryHolderLossesAcrossEpochs object over all 
#         epochs of training.
# - summaryTrainingCurrentEpoch: SummaryHolder object over last epoch of training.
# - summaryTestingAllEpochs: SummaryHolderLossesAcrossEpochs object over all 
#         epochs of testing.
# - summaryTestingCurrentEpoch: SummaryHolder object over last epoch of testing.
def trainWithValidation(loadedConfiguration):
    
    # Read the configuration dictionaries
    configHolder = ConfigHolder.ConfigurationHolder.PrepareConfigHolder(loadedConfiguration)
    
    # Perform training and validation
    kvae, summaryTrainingAllEpochs, summaryTrainingCurrentEpoch, \
           summaryTestingAllEpochs, summaryTestingCurrentEpoch = \
           TrainWithValidationGivenLoadedConfiguration(configHolder.config)

    return kvae, summaryTrainingAllEpochs, summaryTrainingCurrentEpoch, \
           summaryTestingAllEpochs, summaryTestingCurrentEpoch
    
# Function to perform KVAE training.
# INPUTS:
# - loadedConfiguration: path to the configuration file to load.
# OUTPUTS:
# - kvae: trained KVAE model object.
# - summaryTrainingAllEpochs: SummaryHolderLossesAcrossEpochs object over all epochs.
# - summaryTrainingCurrentEpoch: SummaryHolder object over last epoch.
def train(loadedConfiguration):
    
    # Read the configuration dictionaries
    configHolder = ConfigHolder.ConfigurationHolder.PrepareConfigHolder(loadedConfiguration)
    # Perform training
    kvae, summaryTrainingAllEpochs, summaryTrainingCurrentEpoch = trainGivenLoadedConfiguration(configHolder.config)

    return kvae, summaryTrainingAllEpochs, summaryTrainingCurrentEpoch

# Function to perform KVAE testing (over one epoch).
# INPUTS:
# - config: loaded config.
# - kvae
# OUTPUTS:
# - kvae: trained KVAE model object.
# - summaryTestingAllEpochs: SummaryHolderLossesAcrossEpochs object over all epochs.
# - summaryTestingCurrentEpoch: SummaryHolder object over last epoch.
def test(config, kvae):
    
    # Read the configuration dictionaries
    #configHolder = ConfigHolder.ConfigurationHolder.PrepareConfigHolder(loadedConfiguration)
    #config       = configHolder.config
    
    print('Extracting testing data for KVAE')
    testingData, newIndicesTesting, inverseIndexingTesting = \
       DL.LoadDataForKVAESingleTrajectory(dataFile  = config['testing_data_file'], batchSize = config['batch_size'],
                                          image_channels = config['image_channels'])
    
    numberOfDataBatchesTest = testingData.sequences // config['batch_size_test'] # takes floor value
    
    testingData.BringDataToTorch()
       
    kvae, beginEpoch = LoadKVAEwithTrainedKF(kvae, config)
    
    testingDistances  = kvae.FindDistancesFromClusters(testingData.params)
    testingDistances  = torch.from_numpy(testingDistances).to(device)
        
    n = 100000
    
    summaryTestingCurrentEpoch = TestingOneEpochKVAE_D(kvae, config, testingData, n, numberOfDataBatchesTest, testingDistances)
    
    # Save to matlab the values over the current epoch
    summaryTestingCurrentEpoch.BringValuesToMatlab(outputFolder = config['output_folder'], 
                                            dataStructure = 0, 
                                            batchSize = config['batch_size_test'], 
                                            filePrefix = 'VAL_single_epoch_', 
                                            inverseIndexing = inverseIndexingTesting)    
    
    ###########################################################################  
    # PLOT of predicted params (odometry) vs. real one
    
    # Real params
    real_params_for_printing = RD.reshape1DdataWithInverseIndexingAndNoTrajectoryDimension(
            summaryTestingCurrentEpoch.summary['real_params'], inverseIndexingTesting)
    # Predicted params
    predicted_params_min_for_printing = RD.reshape1DdataWithInverseIndexingAndNoTrajectoryDimension(
            summaryTestingCurrentEpoch.summary['predicted_params_min'], inverseIndexingTesting)
    
    # SCATTERPLOT
    PG.plot_predicted_vs_real_states(predicted_params_min_for_printing[:,0:2],
                                     real_params_for_printing[:,0:2],
                                     config['output_folder'] + '/VAL_params_pred_trajectory_' + str(n) + '.png')

    return kvae, summaryTestingCurrentEpoch

###############################################################################
# CODES FOR TRAINING WITH MULTIPLE PARAMETERS

# This creates a 'standard' grid over 4 'standard' KVAE parameters for various
# training attempts.
# OUTPUTS:
# - parameterValues: a list containing in each element the values to consider 
#                  for a particular parameter.
#                  Type: list of 1D arrays.
# - parameterNames: a list containing in each element the names to consider 
#                  for a particular parameter.
#                  Type: list of 1D arrays.
def DefineStandardParameterValuesForMultipleTrainings():
    
    # Where to put the values of the parameters
    parameterValues = []
    # Where to put the names of the parameters
    parameterNames  = []
    
    # Parameters that are important are the following: 
    # 3^4  = 81 combinations
    init_lr           = [0.001, 0.0001]
    init_lr_VAE       = [0.001, 0.0001]
    max_grad_norm_kf  = [0.2, 5]
    max_grad_norm_VAE = [0.2, 5]
    
    # Insert values
    parameterValues.append(init_lr)
    parameterValues.append(init_lr_VAE)
    parameterValues.append(max_grad_norm_kf)
    parameterValues.append(max_grad_norm_VAE)
    # Insert names
    parameterNames.append('init_lr')
    parameterNames.append('init_lr_VAE')
    parameterNames.append('max_grad_norm_kf')
    parameterNames.append('max_grad_norm_VAE')
    
    return parameterValues, parameterNames


# Function to attempt training using a set of different parameters in order to
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
def PerformMultipleKVAE_DTrainings(parametersGrid, config, indexOfGridWhereToBegin = 0):
    
    # Where to put d losses
    d_losses_mean_last_epoch_train = []
    
    # The number of total experiments corresponds to the number of rows of the grid
    numberOfTotalExperiments = parametersGrid.numberOfTests
    # The number of total parameters corresponds to the number of columns of the grid
    numberOfParameters       = parametersGrid.numberOfParameters
    
    # Looping over the experiments
    for i in range(indexOfGridWhereToBegin, numberOfTotalExperiments):
        
        config_new                  = parametersGrid.AssignToDictionaryTheValuesOfGridRow(config, i)
        config_new['output_folder'] = parametersGrid.DefineOutputFolderBasedOnRowOfGrid(config_new['output_folder'], i)
             
        # Call to KVAE training
        kvae, summaryTrainingAllEpochs, summaryCurrentEpoch,  = trainGivenLoadedConfiguration(config_new)
            
        # Finding the mean error of last epoch for d_loss
        total_number_of_epochs      = len(summaryTrainingAllEpochs.summary['d_losses'])
        mean_d_loss_last_epoc_train = summaryTrainingAllEpochs.summary['d_losses'][total_number_of_epochs-1]
        # Append to list
        d_losses_mean_last_epoch_train.append(mean_d_loss_last_epoc_train)
        # Save to MATLAB
        sio.savemat(config['output_folder'] + 'd_losses_mean_last_epoch_train.mat' , 
                    {'d_losses_mean_last_epoch_train': d_losses_mean_last_epoch_train})
        
        del kvae
        del summaryTrainingAllEpochs, summaryCurrentEpoch  
        del mean_d_loss_last_epoc_train
        
        if device.type == "cuda":
            torch.cuda.empty_cache() 
        
    return

# Function as 'PerformMultipleKVAE_DTrainings', but the grid of parameters
# is a 'standard' predefined one, as created by the function 
# 'DefineStandardParameterValuesForMultipleTrainings'.
def PerformMultipleKVAE_DTrainingsWithStandardParametersGrid(config, indexOfGridWhereToBegin = 0):
    
    # Take standard parameters values
    parameterValues, parameterNames = DefineStandardParameterValuesForMultipleTrainings()
    # Create parameters grid
    parametersGrid = TG.TestsGrid(parameterValues, parameterNames)
    # Perform the multiple trainings
    PerformMultipleKVAE_DTrainings(parametersGrid, config)
    
    return

###############################################################################
###############################################################################

class TestsForKVAETraining(unittest.TestCase):
    
    
    def CheckCalculateMeanInAClusterGivenAllEpochStates_1D(self):
        
        z_states = np.array([1,2,3,4,5,6,7,8,9,10])
        clusterAssignments = np.array([0,0,0,1,0,0,0,1,0,1])
        
        expectedMean = (4 + 8 + 10)/3       
        calculatedMean, _ = CalculateMeanInAClusterGivenAllEpochStates(z_states=z_states, 
                                                                    clusterAssignments=clusterAssignments, 
                                                                    clusterIndex=1)       
        self.assertTrue(calculatedMean==expectedMean)
        return
    
    def CheckCalculateMeanInAClusterGivenAllEpochStates_2D(self):
        
        z_states = np.array([[1,-1],[2,-2],[3,-3],[4,-4],[5,-5],[6,-6],[7,-7],[8,-8],[9,-9],[10,-10]])
        clusterAssignments = np.array([0,0,0,1,0,0,0,1,0,1])
        
        expectedMean = [(4 + 8 + 10)/3, (-4 -8 -10)/3]    
        calculatedMean, _ = CalculateMeanInAClusterGivenAllEpochStates(z_states=z_states, 
                                                                    clusterAssignments=clusterAssignments, 
                                                                    clusterIndex=1) 
        self.assertTrue((calculatedMean==expectedMean).all())
        return
    
    @staticmethod
    def PerformAllTests():
        
        TestForKVAETraining = TestsForKVAETraining()
        TestForKVAETraining.CheckCalculateMeanInAClusterGivenAllEpochStates_1D()
        TestForKVAETraining.CheckCalculateMeanInAClusterGivenAllEpochStates_2D()
        print('All tests have been successfully performed')
        return
        
def main():
    TestsForKVAETraining.PerformAllTests()
    
main() 
        