
import numpy as np

import torch
import torch.nn as nn

from KVAE_models_and_utils_codes import VAE
from KVAE_models_and_utils_codes import Model
from KVAE_models_and_utils_codes import Distance_utils  as d_utils
import torch.nn.functional as F
from KVAE_models_and_utils_codes import ClusteringGraph as CG

from KVAE_models_and_utils_codes import p_filter

from ConfigurationFiles import Config_GPU as ConfigGPU

import unittest

###############################################################################
# ---------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)

###############################################################################
# ------------------  KalmanVariationalAutoencoder   --------------------------
###############################################################################

###############################################################################
# Exception classes

class InconsistentDimensionsOfLatentStates(Exception):

    def __init__(self, dim_a, dim_z):
        self.dim_a = dim_a
        self.dim_z = dim_z    
    def __str__(self):
        message = 'State z cannot be of bigger dimensionality than state a. ' + \
                  'But assigned dimensions where {} (for a) and {} (for z).'.format(self.dim_a, self.dim_z)
        return message
    
###############################################################################
# KVAE class

class KalmanVariationalAutoencoder(Model.Model):
    """ This class defines functions to build, train and evaluate Kalman Variational Autoencoders
    """
    
    # Function to initialize the VAE subcomponent of the KVAE.
    # INPUTS:
    # - config: configuration file
    # - trainingData: data of training.
    def BuildVAESubcomponent(self, config, trainingData):
        
        num_filters = [int(f) for f in config['dim_filters'].split(',')]
        # Create VAE
        self.baseVAE = VAE.VAE(z_dim          = config['dim_a'], 
                               image_channels = config['image_channels'], 
                               image_size     = [config['dimension_x'],config['dimension_y']], 
                               dim_filters    = num_filters, 
                               kernel         = config['filter_size'], 
                               stride         = config['stride']).to(device)
        
        self.baseVAE.print()
        self.baseVAE.PrintVAELayers(config['image_channels']) 
        
        return 
    
    # Function to initialize the Kalman Filter subcomponent of the KVAE.
    # INPUTS:
    # - config: configuration file
    # - clusterGraph: clustering object of odometry
    # - trainingData: data of training.
    # - sequence_length: length of sequences
    def BuildKFSubcomponent(self, config, clusterGraph, trainingData, sequence_length = None):
        
        if sequence_length == None:
            sequence_length = config['sequence_length']
        
        self.kf = p_filter.KalmanFilter(dim_z            = config['dim_z'],
                                        dim_y            = config['dim_a'],
                                        num_clusters     = clusterGraph.num_clusters,
                                        init_kf_matrices = config['init_kf_matrices'],
                                        init_cov         = config['init_cov'],
                                        batch_size       = config['batch_size'],
                                        noise_transition = config['noise_transition'],
                                        noise_emission   = config['noise_emission'], 
                                        sequence_length  = sequence_length).to(device)        
        return 
    
    # Function to initialize the VAE and Kalman Filter subcomponent of the KVAE.
    # INPUTS:
    # - config: configuration file
    # - clusterGraph: clustering object of odometry
    # - trainingData: data of training.
    # - sequence_length: length of sequences
    def BuildSubcomponentsOfKVAE(self, config, trainingData, clusterGraph, sequence_length = None):
        
        # Creating the base VAE
        print('Loading VAE subcomponent')
        self.BuildVAESubcomponent(config, trainingData)
        # Creating the kf
        print('Loading KF subcomponent')
        self.BuildKFSubcomponent(config, clusterGraph, trainingData, sequence_length)
        
        return 
    
    def _ExtractStatesDimension(self, config):
        
        self.dim_a = config['dim_a']
        self.dim_z = config['dim_z']
        
        return
    
    def _FindParamsDimension(self, config):
        
        self.paramsDimension = 2
        
        return
    
    def _FindClusteringDimension(self, config):
        
        self.clusteringDimension = 4
        '''
        if self.z_meaning == 0: # odometry -> pick velocity only
            self.paramsDimension = 2
        elif self.z_meaning == 1: # theta + vel norm
            self.paramsDimension = 2
        elif self.z_meaning == 2: # position + theta + vel norm (but position does not count)
            self.paramsDimension = 4
        elif self.z_meaning == 10: # distance
            self.paramsDimension = 2   
        '''        
        return 
    
    # Checking for wrong KVAE structure
    def CheckForExceptionsOnStructure(self):
        
        # Check that the first two dimensions at the end of convolutions did not go below zero
        if self.dim_a < self.dim_z:
            raise InconsistentDimensionsOfLatentStates(self.dim_a, self.dim_z)
        return
    
    # Initialization
    # Inputs:
    # - config: configuration dictionary.
    # - clusterGraph: object of class ClusterGraph, to use to pass the clustering
    #                 information.
    # - trainingData: data of training 
    # - sequence_length (typically not present in training; used in testing to set it to 1)
    def __init__(self, config, clusterGraph, trainingData = None, sequence_length = None):
        super(KalmanVariationalAutoencoder, self).__init__()
        
        #######################################################################     
        self._ExtractStatesDimension(config)
        self._FindParamsDimension(config)
        self._FindClusteringDimension(config)
        #######################################################################
        # Check structure
        self.CheckForExceptionsOnStructure()
        #######################################################################
        # Building the two main subcomponents of KVAE: vae and kf
        # The clusterGraph is included separately because it needs to be trained
        # before. Instead, kf and vae are created here.
        self.BuildSubcomponentsOfKVAE(config, trainingData, clusterGraph, sequence_length)
        #######################################################################
        # Graph of clusters
        self.clusterGraph = clusterGraph
        #######################################################################
        # configuration information for the 
        # model (params of training, dimension of networks etc.)
        self.config = config    
        #######################################################################        
        self.skewness  = config['skewness']       
        self.z_meaning = config['z_meaning']     
        #######################################################################        
        self.lb_vars = None
        return
        
    ###########################################################################
    # DISTANCE FROM CLUSTERS functions
    
    # Function to find the distances of a set of sequences of data
    # odometry from the clusters of the odometry clustering of the KVAE.
    # This is to be used IN TESTING. In this way, the training distances 
    # memorized in the KVAE are not modified.
    # - clusterParamsData: values of odometry of sequences.
    def FindDistancesFromClusters(self, clusterParamsData):
        
        reshaped = False
        
        if clusterParamsData.ndim == 4:
            
            reshaped = True
            
            numTrajectories          = clusterParamsData.shape[0]
            numSequencesInTrajectory = clusterParamsData.shape[1]
            sequenceLength           = clusterParamsData.shape[2]
            dimensionParams          = clusterParamsData.shape[3]
            
            if type(clusterParamsData)   == np.ndarray:
                clusterParamsData = np.reshape(clusterParamsData, 
                                                   (numTrajectories*numSequencesInTrajectory, sequenceLength, dimensionParams))
            elif type(clusterParamsData) == torch.Tensor:
                clusterParamsData = torch.reshape(clusterParamsData, 
                                                   (numTrajectories*numSequencesInTrajectory, sequenceLength, dimensionParams))
        
        # Define an observation covariance for each 
        dim_sequences     = clusterParamsData.shape[2]
        r1                = 1e-18
        
        if type(clusterParamsData)   == np.ndarray:
            obsCovariance = np.eye(dim_sequences)*r1
        elif type(clusterParamsData) == torch.Tensor:
            obsCovariance = (torch.eye(dim_sequences)*r1).to(device)
        
        # Extract the Bhattacharya distances of the datapoints from the 
        # clusters of the graph
        distances    = self.clusterGraph.FindBhattaDistancesFromSequencesToClusters(clusterParamsData, obsCovariance)
        
        if reshaped == True:
            
            numOfClusters = distances.shape[2]
            
            distances     = np.reshape(distances, (numTrajectories, numSequencesInTrajectory, sequenceLength, numOfClusters))
        
        return distances
        
    ###########################################################################
    # Skenewss fixing
        
    # Function to increment the skewness value
    # Inputs:
    # - skewnessIncrement: of how much to increment the skewness from current value
    def IncrementSkewness(self, skewnessIncrement):
        
        self.skewness = self.skewness + skewnessIncrement
        
        return
    
    # Function to fix a new value of skewness
    # Inputs:
    # - skewness: new value for skewness
    def FixSkewness(self, skewness):
        
        self.skewness = skewness
        
        return
    
    ###########################################################################
    # VAE INTERFACE:
    # this is necessary because VAE considers single images, whereas KVAE 
    # considers SEQUENCES

    # FLATTEN BATCH OF IMAGES
    # Unite batch and sequence dimension
    # Original dimension:
    # [batch number, color_channel, sequences_length, imageDimensionX, imageDimensionY]
    # New dimension:
    # [batch number, color_channel*sequences_length, imageDimensionX, imageDimensionY]
    # Inputs:
    # - currentImagesBatch: batch of sequences of images to flatten
    # Outputs:
    # - currentDataBatchFlattenBatchSequence: flattened batch
    def FlattenBatchSequence(self, currentImagesBatch):
        
        currentImagesBatch = torch.permute(currentImagesBatch, (0,2,1,3,4))
        
        currentDataBatchFlattenBatchSequence = torch.reshape(currentImagesBatch, (-1 , 
                                                                        currentImagesBatch.shape[2], # image channels
                                                                        currentImagesBatch.shape[3], # imageSizeX
                                                                        currentImagesBatch.shape[4])) # imageSizeY
        
        return currentDataBatchFlattenBatchSequence
    
    # UNFLATTEN BATCH OF IMAGES
    # Bring all to original dimensions
    # From:
    # [batch number, color_channel*sequences_length, imageDimensionX, imageDimensionY]
    # To:
    # [batch number, color_channel, sequences_length, imageDimensionX, imageDimensionY]
    # Inputs:
    # - currentImagesBatch: original images, from which to retrieve dimensions
    # - currentDataBatchFlattenBatchSequence: reconstructed flattened batch of sequences of images
    # Outputs:
    # - reconstructedImagesBatchUnflattened: reconstructed unflattened batch of sequences of images
    def UnflattenBatchSequence(self, currentImagesBatch, reconstructedImagesBatch):
        
        reconstructedImagesBatchUnflattened          = reconstructedImagesBatch.view(currentImagesBatch.shape[0], # batch number
                                                                                     currentImagesBatch.shape[2], # sequence len
                                                                                     currentImagesBatch.shape[1], # image channels 
                                                                                     currentImagesBatch.shape[3], # imageSizeX
                                                                                     currentImagesBatch.shape[4]) # imageSizeY     
        
        reconstructedImagesBatchUnflattened          = torch.permute(reconstructedImagesBatchUnflattened, (0,2,1,3,4))   
        
        return reconstructedImagesBatchUnflattened
    
    # Similar to "UnflattenBatchSequence", but to unflatten the bottleneck states
    # Inputs:
    # - state: could be a_seq, a_mu or a_var
    # - batchSize: batch size
    # - sequenceLength: length of sequences in batch
    # Outputs:
    # - unflattenedState: Unflattened state
    def UnflattenBottleneckStates(self, state, batchSize, sequenceLength):
        
        unflattenedState = state.view(batchSize,      # batch number
                                      sequenceLength, # sequence len
                                      state.shape[1]) # state dimension
        return unflattenedState
        
    # Function to perform encoding and decoding of images in batch.
    # Bottleneck params and reconstructed images are extracted
    def CallVAEEncoderAndDecoderOverBatchSequence(self, currentImagesBatch):
        
        # FLATTEN
        currentDataBatchFlattenBatchSequence         = self.FlattenBatchSequence(currentImagesBatch)
    
        # ENCODE and DECODE
        # Call the encoder and decoder of the VAE
        reconstructedImagesBatch, a_seq, a_mu, a_var = self.baseVAE(currentDataBatchFlattenBatchSequence)
        
        # UNFLATTEN IMAGES
        reconstructedImagesBatchUnflattened          = self.UnflattenBatchSequence(currentImagesBatch, reconstructedImagesBatch)
        
        # UNFLATTEN STATES
        a_seqUnflattened = self.UnflattenBottleneckStates(a_seq, currentImagesBatch.shape[0], currentImagesBatch.shape[2])
        a_muUnflattened  = self.UnflattenBottleneckStates(a_mu , currentImagesBatch.shape[0], currentImagesBatch.shape[2])
        a_varUnflattened = self.UnflattenBottleneckStates(a_var, currentImagesBatch.shape[0], currentImagesBatch.shape[2])
                        
        return  reconstructedImagesBatchUnflattened, a_seqUnflattened, a_muUnflattened, a_varUnflattened
    
    ###########################################################################
    # Alpha vector
    
    # Finding the alpha vector from the distanced from each odometry cluster,
    # given a skew value.
    # INPUTS:
    # - distances: distances from each cluster,
    # - skew: temperature value.
    # - alphaPrev: alpha of previous time instant.
    def alphaDistProbGivenSkew(self, distances, skew, alphaPrev = None, 
                               timeInClustersInput = None, previousMaxAlphasInput = None):
        
        probabilities = d_utils.CalculateProbabilitiesFromDistances(distances, skew)
            
        return probabilities

    # Finding the alpha vector from the distanced from each odometry cluster,
    # given the skew value of the KVAE object.
    def alphaDistProb(self, distances, alphaPrev = None, timeInClusters = None, previousMaxAlphas = None):
        
        skew = self.skewness
        return self.alphaDistProbGivenSkew(distances, skew, alphaPrev, timeInClusters, previousMaxAlphas)
    
    ###########################################################################
    # FUNCTIONS TO CALCULATE THE LOSSES
    
    # A) transition and emission losses
    
    @staticmethod
    def ExtractStateZwithCovarianceFromBackwardStates(backward_states):
        
        mu_smooth      = torch.squeeze(backward_states[0])
        Sigma_smooth   = backward_states[1]
        
        return mu_smooth, Sigma_smooth
    
    def CalculateEmissionLossProbabilities(self, statesZ, covarianceStatesZ, C):
        
        ## Emission distribution \prod_{t=1}^T p(y_t|z_t)
        # We need to evaluate N(y_t; Cz_t, R). We write it as N(y_t - Cz_t; 0, R)
        statesAFromZ                    = torch.reshape(torch.matmul(C, torch.unsqueeze(statesZ, 3)), [-1, self.kf.dim_y])
        covarianceStatesAFromZ          = torch.matmul(torch.matmul(C, covarianceStatesZ),C.permute(0, 1,3, 2)) 
        covarianceStatesAFromZ_reshaped = torch.reshape(covarianceStatesAFromZ, [-1, self.kf.dim_y, self.kf.dim_y])

        statesAFromVAE                  = torch.reshape(self.kf.y, [-1, self.kf.dim_y])
        
        return statesAFromZ, covarianceStatesAFromZ_reshaped, statesAFromVAE
    
    def CalculateTransitionLossProbabilities(self, statesZ, covarianceStatesZ, A, B):
        
        ## Transition distribution \prod_{t=2}^T p(z_t|z_{t-1}, u_{t})
        # We need to evaluate N(z_t; Az_tm1 + Bu_t, Q), where Q is the same for all the elements
        nextZThroughA = torch.reshape(torch.matmul(A[:, :-1], torch.unsqueeze(statesZ[:, :-1], 3)), [-1, self.kf.dim_z])
        #Az_tm1_0       = torch.reshape(torch.matmul(A[0,0], torch.unsqueeze(mu_smooth_0, 1)), [-1, self.kf.dim_z])
        # Remove the first input as our prior over z_1 does not depend on it
        Bu_t          = torch.reshape(B[:, :-1], [-1, self.kf.dim_z])
        
        nextZThroughA  = nextZThroughA + Bu_t
        
        '''
        # next time instant
        if self.kf.consider_A == True and self.kf.consider_B == True:
            nextZThroughA  = nextZThroughA + Bu_t
        elif self.kf.consider_A == True:
            nextZThroughA  = nextZThroughA
        elif self.kf.consider_B == True:
            nextZThroughA  = torch.reshape(statesZ[:, :-1], [-1, self.kf.dim_z]) + Bu_t
        '''
            
        # note that it is taken from time 1 and not time 0: the first time 
        # instant is obviously jumped
        nextZFromSmoothing = torch.reshape(statesZ[:, 1:, :], [-1, self.kf.dim_z])
        
        return nextZThroughA, nextZFromSmoothing
    
    def CalculateBhattacharyaEmissionLoss(self, backward_states, C):
        
        # Extract the smoothed Z states (mu_smooth in original code), with its
        # covariances (sigma_smooth), from the backward states
        statesZ, covarianceStatesZ = \
           KalmanVariationalAutoencoder.ExtractStateZwithCovarianceFromBackwardStates(backward_states)
        # Calculate parts of distributions necessary for loss calculation
        statesAFromZ, covarianceStatesAFromZ_reshaped, statesAFromVAE = \
           self.CalculateEmissionLossProbabilities(statesZ, covarianceStatesZ, C)
            
        # Calculate the loss for each value
        bhattacharya_distance_emission_sum = 0
        for j in range(statesAFromZ.shape[0]):
            # For current value
            bhattacharya_distance_current = d_utils.CalculateBhattacharyyaDistanceTorch(statesAFromZ[j]  , torch.diag(covarianceStatesAFromZ_reshaped[j]),
                                                                                        statesAFromVAE[j], torch.diag(self.kf.R)) 
            # Add to sum
            bhattacharya_distance_emission_sum += bhattacharya_distance_current    
            
        return bhattacharya_distance_emission_sum
    
    def CalculateKLDEmissionLoss(self, backward_states, C):
        
        # Extract the smoothed Z states (mu_smooth in original code), with its
        # covariances (sigma_smooth), from the backward states
        statesZ, covarianceStatesZ = \
           KalmanVariationalAutoencoder.ExtractStateZwithCovarianceFromBackwardStates(backward_states)
        # Calculate parts of distributions necessary for loss calculation
        statesAFromZ, covarianceStatesAFromZ_reshaped, statesAFromVAE = \
           self.CalculateEmissionLossProbabilities(statesZ, covarianceStatesZ, C)
            
        # Calculate the loss for each value
        kld_distance_emission_sum = 0
        for j in range(statesAFromZ.shape[0]):
            # For current value
            kld_distance_current = d_utils.gau_klTorch(statesAFromZ[j]  , torch.diag(covarianceStatesAFromZ_reshaped[j]),
                                                       statesAFromVAE[j], torch.diag(self.kf.R)) 
            # Add to sum
            kld_distance_emission_sum += kld_distance_current    
            
        return kld_distance_emission_sum
    
    def CalculateBhattacharyaTransitionLoss(self, backward_states, A, B):
        
        # Extract the smoothed Z states (mu_smooth in original code), with its
        # covariances (sigma_smooth), from the backward states
        statesZ, covarianceStatesZ = \
           KalmanVariationalAutoencoder.ExtractStateZwithCovarianceFromBackwardStates(backward_states)
        # Reshape covarianceStatesZ so to use it in the calculation of the distance
        covarianceStatesZ_reshaped = torch.reshape(covarianceStatesZ[:,:-1,:,:], [-1, self.kf.dim_z, self.kf.dim_z])
        # Calculate parts of distributions necessary for loss calculation
        nextZThroughA, nextZFromSmoothing = self.CalculateTransitionLossProbabilities(statesZ, covarianceStatesZ, A, B)
        
        # Calculate the loss for each value
        bhattacharya_distance_transition_sum = 0
        for j in range(nextZThroughA.shape[0]):
            # For current value
            bhattacharya_distance_current = d_utils.CalculateBhattacharyyaDistanceTorch(nextZThroughA[j]     , torch.diag(covarianceStatesZ_reshaped[j]),
                                                                                        nextZFromSmoothing[j], torch.diag(self.kf.Q))
            # Add to sum
            bhattacharya_distance_transition_sum += bhattacharya_distance_current
        
        return bhattacharya_distance_transition_sum, nextZThroughA, nextZFromSmoothing
    
    def CalculateKLDTransitionLoss(self, backward_states, A, B):
        
        # Extract the smoothed Z states (mu_smooth in original code), with its
        # covariances (sigma_smooth), from the backward states
        statesZ, covarianceStatesZ = \
           KalmanVariationalAutoencoder.ExtractStateZwithCovarianceFromBackwardStates(backward_states)
        # Reshape covarianceStatesZ so to use it in the calculation of the distance
        covarianceStatesZ_reshaped = torch.reshape(covarianceStatesZ[:,:-1,:,:], [-1, self.kf.dim_z, self.kf.dim_z])
        # Calculate parts of distributions necessary for loss calculation
        nextZThroughA, nextZFromSmoothing = self.CalculateTransitionLossProbabilities(statesZ, covarianceStatesZ, A, B)
        
        # Calculate the loss for each value
        kld_distance_transition_sum = 0
        for j in range(nextZThroughA.shape[0]):
            # For current value
            kld_distance_current = d_utils.gau_klTorch(nextZThroughA[j]     , torch.diag(covarianceStatesZ_reshaped[j]),
                                                       nextZFromSmoothing[j], torch.diag(self.kf.Q))
            # Add to sum
            kld_distance_transition_sum += kld_distance_current
        
        return kld_distance_transition_sum, nextZThroughA, nextZFromSmoothing
    
    # B) Filtering vs. smoothing loss
    
    def CalculateSmoothingFilteringEqualityLoss(self, backward_states, mu_t, mu_var, A, B):
        
        # Extract the smoothed Z states (mu_smooth in original code), with its
        # covariances (sigma_smooth), from the backward states
        statesZ, covarianceStatesZ = \
           KalmanVariationalAutoencoder.ExtractStateZwithCovarianceFromBackwardStates(backward_states)
        
        mu_t = torch.swapaxes(mu_t, 0, 1)
           
        smoothingFilteringEqualityLoss = torch.mean(torch.mean(torch.mean(torch.abs(statesZ-mu_t))))
        
        return smoothingFilteringEqualityLoss
    
    # C) Triplet loss
    

    
    ###########################################################################
    # In case of calling encoder and decoder separately! 
    
    def CallVAEEncoderOverBatchSequence():
        dummy = 0
        
    def CallVAEDecoderOverBatchSequence():
        dummy = 0
        
    ###########################################################################
        
    @staticmethod
    def NormalizeData(data, maxValue, minValue):
        
        rangeValue     = maxValue - minValue
        normalizedData = (data - minValue)/rangeValue
        
        return normalizedData
    
    ###########################################################################
    # Smoothing/Filtering functions
        
    # Performing KVAE smoothing over a sequence of data
    # INPUTS:
    # - currentImagesBatch: images of current batch
    # - curr_distance: distance values from clusters of odometry, for datapoints
    #   in the current batch.
    # OUTPUTS:
    # - reconstructedImagesBatch: reconstructed images (corresponding to 
    #   'currentImagesBatch') by the VAE,
    # - a_seq: states 'a', sampled,
    # - a_mu: states 'a', mean
    # - a_var: states 'a', mean
    # - smooth: states 'z'. This is a tuple containing the mean and the covariance.
    # - A, B, C: matrices of KVAE
    # - alpha_plot: value of 'alpha' (probabilities of clusters)
    def PerformKVAESmoothingOverBatchSequence(self, currentImagesBatch,curr_distance):
        
        ################################# VAE #################################
        
        # ENCODE AND DECODE
        reconstructedImagesBatch, a_seq, a_mu, a_var = self.CallVAEEncoderAndDecoderOverBatchSequence(currentImagesBatch)
        
        ################################# KF ##################################
        
        self.kf.initializeCallToKalmanFilter(dist=curr_distance, 
                                             y=a_seq, 
                                             alphaDistProb = self.alphaDistProb)
        
        smooth, A, B, C, alpha_plot, mu_preds, Sigma_preds, mu_filts, Sigma_filts = self.kf.smooth()

        return reconstructedImagesBatch, a_seq, a_mu, a_var, smooth, A, B, C, alpha_plot, mu_preds, Sigma_preds, mu_filts, Sigma_filts
    
    # Performing KVAE filtering over a sequence of data
    # INPUTS:
    # - currentImagesBatch: images of current batch
    # - curr_distance: distance values from clusters of odometry, for datapoints
    #   in the current batch.
    # OUTPUTS:
    # - reconstructedImagesBatch: reconstructed images (corresponding to 
    #   'currentImagesBatch') by the VAE,
    # - a_seq: states 'a', sampled,
    # - a_mu: states 'a', mean
    # - a_var: states 'a', mean
    # - filter: states 'z'. This is a tuple containing the mean and the covariance.
    # - A, B, C_filter: matrices of KVAE
    # - alpha_plot: value of 'alpha' (probabilities of clusters)
    def PerformKVAEFilteringOverBatchSequence(self, currentImagesBatch,curr_distance):
        
        ################################# VAE #################################
        
        # ENCODE AND DECODE
        reconstructedImagesBatch, a_seq, a_mu, a_var = self.CallVAEEncoderAndDecoderOverBatchSequence(currentImagesBatch)
        
        ################################# KF ##################################
        
        self.kf.initializeCallToKalmanFilter(dist=curr_distance, 
                                             y=a_seq, 
                                             alphaDistProb = self.alphaDistProb)
     
        mu_pred, Sigma_pred, filter, A, B, C_filter, alpha_values = self.kf.filter()
        self.mu_pred    = mu_pred.clone()
        self.Sigma_pred = Sigma_pred.clone()

        return reconstructedImagesBatch, a_seq, a_mu, a_var, filter, alpha_values, A, B, C_filter
    
    ###########################################################################
    # Functions for preparing input to KVAE, for training and testing.
    
    # This is a function for preparing the input data to be given as input to the KVAE,
    # batch by batch.
    # The way in which the data is extracted from the 'trainingData' structure 
    # depends on how the data has been structured itself.
    def PrepareInputsForKVAE(self, data, distances, batchSize, currentBatchNumber, image_channels):
        
        # Extract from data
        currentImagesBatch, currentControlsBatch, currentOdometryBatch, currentParamsBatch, currentDistancesBatch = \
           KalmanVariationalAutoencoder.ExtractBatchInputsDataStructure4D(data, distances, currentBatchNumber, 
                                                                          batchSize, image_channels)         
        # If there is no color channel, add one after batch size
        if len(currentImagesBatch.shape) == 4:
            currentImagesBatch = torch.unsqueeze(currentImagesBatch, 1)

        return currentImagesBatch, currentControlsBatch, currentOdometryBatch, \
               currentParamsBatch, currentDistancesBatch
               
    ###########################################################################
    # Functions for extracting batches. These are used in 'PrepareInputsForKVAETraining'
    # and in 'PrepareInputsForKVAETesting'.
    
    @staticmethod 
    def ExtractBatchInputsDataStructure4DWithoutDistances(data, currentBatchNumber, batchSize, image_channels):
        
         # Select slice corresponding to batch
        slc = slice(currentBatchNumber * batchSize, (currentBatchNumber + 1) * batchSize)
        
        if image_channels == 1:
            # Select the batch for images
            currentImagesBatch    = data.images[slc].to(device)
            
        elif image_channels == 3:
            # Select the batch for images
            currentImagesBatch    = data.images[:,slc].to(device)            
            # invert color and batch dimensions
            currentImagesBatch = torch.swapaxes(currentImagesBatch, 0, 1)
            
        # Also for controls, odometry and distances
        currentControlsBatch  = data.controls[slc].to(device)
        currentOdometryBatch  = data.odometry[slc].to(device)
        currentParamsBatch    = data.params[slc].to(device)
        
        return currentImagesBatch, currentControlsBatch, currentOdometryBatch, currentParamsBatch
        
    @staticmethod
    def ExtractBatchInputsDataStructure4D(data, distances, currentBatchNumber, batchSize, image_channels):
        
        # Select slice corresponding to batch
        slc = slice(currentBatchNumber * batchSize, (currentBatchNumber + 1) * batchSize)
        
        currentImagesBatch, currentControlsBatch, currentOdometryBatch, currentParamsBatch = \
           KalmanVariationalAutoencoder.ExtractBatchInputsDataStructure4DWithoutDistances(data, currentBatchNumber, batchSize, image_channels)
        
        currentDistancesBatch = distances[slc].to(device)

        return currentImagesBatch, currentControlsBatch, currentOdometryBatch, \
           currentParamsBatch, currentDistancesBatch

    ###########################################################################
    # TRAINING functions
    
    # Function to perform the training of the KVAE
    # What this function does is the following:
    # - takes the current image and inputs it to the VAE, performing encoding and
    #   decoding.
    # - takes the bottleneck from the VAE and performs Kalman Smoothing on it
    #   for the batch along the sequences in each batch
    # - Calculates the losses related to VAE and to Kalman Filtering.
    def PerformKVAETrainingOverBatchSequence(self, currentImagesBatch,curr_distance,currentClusterParamsBatch):
        
        ########################## KVAE SMOOTHING #############################
        # Training of KVAE is done with call to smoothing function
        reconstructedImagesBatch, a_seq, a_mu, a_var, smooth, A, B, C, alpha_plot = \
           self.PerformKVAESmoothingOverBatchSequence(currentImagesBatch,curr_distance)
           
        ################################# LOSSES ##############################
        # Calculate losses
        losses, z_smooth, predicted_params = \
           self.CalculateKVAELoss(a_mu, a_var, reconstructedImagesBatch, currentImagesBatch, 
                                  smooth, A, B, C, currentClusterParamsBatch)
        
        return reconstructedImagesBatch, a_seq, a_mu, a_var, losses, \
            z_smooth, smooth, A ,B ,C ,alpha_plot, predicted_params
            
    def PerformVanillaVATTrainingOverBatchSequence(self, currentImagesBatch):
        
        ################################# VAE #################################        
        # ENCODE AND DECODE
        reconstructedImagesBatch, a_seq, a_mu, a_var = self.CallVAEEncoderAndDecoderOverBatchSequence(currentImagesBatch)       
        # FIND LOSSES of VAE
        KLDLoss, ReconstructionLoss, vaeLoss = self.CalculateVAELoss(a_mu, a_var, reconstructedImagesBatch, currentImagesBatch)        
        return reconstructedImagesBatch, a_seq, a_mu, a_var, KLDLoss, ReconstructionLoss, vaeLoss

    ###########################################################################
    # PRINT what the KVAE contains
            
    def print(self):
        
        # Print Graph info
        self.clusterGraph.print()         
        # Print VAE info
        self.baseVAE.print()        
        return
    
    ###########################################################################
    # For triplet loss (NOT USED ANYMORE)
    
    # Finds the connected clusters in a batch of elements w.r.t. the 
    # cluster of a datapoint.
    @staticmethod
    def FindConnectedClustersInBatch(transitionMat, clustersInBatch, currentAssignment):
        
        transitionMatrixRow = transitionMat[currentAssignment,:]
        transitionMatrixRowWithZeroedClusterItself = transitionMatrixRow.clone()
        transitionMatrixRowWithZeroedClusterItself[currentAssignment] = 0
        # Connected cluster to this cluster
        connectedClusters = torch.where(transitionMatrixRowWithZeroedClusterItself!=0)[0]
        # Look for connected in this batch                    
        connectedClustersInBatch = np.intersect1d(clustersInBatch.detach().cpu().numpy(), connectedClusters.detach().cpu().numpy())
        return connectedClustersInBatch
    
    # Finds the unconnected clusters in a batch of elements w.r.t. the 
    # cluster of a datapoint.
    @staticmethod
    def FindUnconnectedClustersInBatch(transitionMat, clustersInBatch, currentAssignment):
        
        transitionMatrixRow = transitionMat[currentAssignment,:]
        # Unconnected cluster to this cluster
        unconnectedClusters = torch.where(transitionMatrixRow==0)[0]
        # Look for unconnected clusters in this batch                    
        unconnectedClustersInBatch = np.intersect1d(clustersInBatch.detach().cpu().numpy(), unconnectedClusters.detach().cpu().numpy())
        return unconnectedClustersInBatch
    
    @staticmethod
    def SelectPointInBatchBelongingToClustersSubset(clustersSubset, clusterAssignments, statesZ):
        
        # ... otherwise, randomly pick one cluster in the batch belonging to the subset
        randomClusterInBatch = int(np.random.choice(clustersSubset))
        # Points belonging to those clusters
        indicesOfPointsBelongingToSubset = torch.where(clusterAssignments == randomClusterInBatch)
        numberPointsOfSelectedCluster = len(indicesOfPointsBelongingToSubset[0])
        # randomly select one of the points
        indexSelectedPoint = np.random.choice(numberPointsOfSelectedCluster)
        indexSelectedPointInOriginalVector_batch = indicesOfPointsBelongingToSubset[0][indexSelectedPoint]
        indexSelectedPointInOriginalVector_point = indicesOfPointsBelongingToSubset[1][indexSelectedPoint]
        # Take the points
        selectedPoint = statesZ[indexSelectedPointInOriginalVector_batch,indexSelectedPointInOriginalVector_point,:]        
        return selectedPoint
    
    def CalculateTripletLossZFromState(self, statesZ, alpha_plot):
        
        # Transition matrix
        transitionMat = self.clusterGraph.transitionMat
        
        # Calculate cluster assignments of the points
        #clusterAssignments = d_utils.FindHighestValuesAlongDimension(alpha_plot, 2)
        clusterAssignments = d_utils.SampleClassFromDiscreteDistribution3D(alpha_plot)
        # Clusters in the batch
        clustersInBatch = torch.unique(clusterAssignments)
        numberOfClustersInBatch = len(clustersInBatch)
        
        # We initialize it to zero, in case there is not a pair of
        # connected and not-connected cluster in the transition matrix
        tripletLossZ = torch.zeros(1).to(device)

        # Check if we have only one or two clusters, if yes exit with null loss
        if numberOfClustersInBatch == 1 or numberOfClustersInBatch == 2:  
            print('No conditions for triplet loss calculation')
            tripletLossZ = torch.zeros(1).to(device)
        else:
            for i in range(clusterAssignments.shape[0]):
                for j in range(clusterAssignments.shape[1]):
                    
                    currentPoint = statesZ[i,j,:]
                    currentAssignment = int(clusterAssignments[i,j])
                    # Find connected and unconnected clusters in batch
                    connectedClustersInBatch = KalmanVariationalAutoencoder.FindConnectedClustersInBatch(
                            transitionMat, clustersInBatch, currentAssignment)
                    unconnectedClustersInBatch = KalmanVariationalAutoencoder.FindUnconnectedClustersInBatch(
                            transitionMat, clustersInBatch, currentAssignment)
                    # If there is no intersection in one of the two, continue to next i,j ...
                    if len(connectedClustersInBatch) == 0 or len(unconnectedClustersInBatch) == 0:
                        continue
                    else:
                        # Extract a random point of the connected and unconnect cluster
                        connectedPoint = KalmanVariationalAutoencoder.SelectPointInBatchBelongingToClustersSubset(
                                connectedClustersInBatch, clusterAssignments, statesZ)
                        unconnectedPoint = KalmanVariationalAutoencoder.SelectPointInBatchBelongingToClustersSubset(
                                unconnectedClustersInBatch, clusterAssignments, statesZ)
                        # FINALLY CALCULATE TRIPLET LOSS
                        distanceWithConnectedPoint = torch.mean(torch.mean(torch.mean(torch.abs(currentPoint - connectedPoint))))
                        distanceWithUnconnectedPoint = torch.mean(torch.mean(torch.mean(torch.abs(currentPoint - unconnectedPoint))))
                        #distanceWithConnectedPoint = F.mse_loss(currentPoint, connectedPoint,size_average=False)
                        #distanceWithUnconnectedPoint = F.mse_loss(currentPoint, unconnectedPoint,size_average=False)
                        currentTripletLoss = distanceWithConnectedPoint - distanceWithUnconnectedPoint
                        # Sum the triplet losses and continue
                        tripletLossZ += currentTripletLoss   
                        
        return tripletLossZ
    
    def CalculateTripletLossZ(self, backward_states, alpha_plot):
        
        # Extract the smoothed Z states (mu_smooth in original code), with its
        # covariances (sigma_smooth), from the backward states
        statesZ, covarianceStatesZ = \
           KalmanVariationalAutoencoder.ExtractStateZwithCovarianceFromBackwardStates(backward_states)          
        # Triplet loss        
        tripletLossZ = self.CalculateTripletLossZFromState(statesZ, alpha_plot)        
        return tripletLossZ
    
    ###########################################################################
    # For further linearization
    
    
###############################################################################
# Tests classes for checking secondary parts of the code.
        

class TestsKVAE(unittest.TestCase):
    
    def _InitializeDummyKVAE(self):
        
        config = dict()
        config['dim_z'] = 2
        config['dim_a'] = 32
        config['dim_filters'] = "32,64,128"
        config['init_kf_matrices'] = 0.05
        config['init_cov'] = 0.2
        config['batch_size'] = 3
        config['noise_transition'] = 0.2
        config['noise_emission'] = 0.2
        config['image_channels'] = 3
        config['dimension_x'] = 64
        config['dimension_y'] = 64
        config['filter_size'] = 3
        config['stride'] = 2
        config['sequence_length'] = 5
        config['skewness'] = 10  
        config['z_meaning'] = 0
        
        clusterGraph = CG.ClusteringGraph(z_dim = 2)
        clusterGraph.num_clusters = 3
        
        self.kvae = KalmanVariationalAutoencoder(config, clusterGraph)    
        
        return
    
    def CheckThatTripletLossZIsZeroWhenAllDataBelongsToTheSameCluster(self):
        
        self._InitializeDummyKVAE()
        
        batch_size = self.kvae.config['batch_size']
        dim_z = self.kvae.config['dim_z']
        sequence_length = self.kvae.config['sequence_length']
        num_clusters = self.kvae.clusterGraph.num_clusters
        
        statesZ    = torch.rand(batch_size,sequence_length,dim_z)
        alpha_plot = torch.ones(batch_size,sequence_length,num_clusters)
        self.kvae.clusterGraph.transitionMat = torch.eye(num_clusters)
        
        tripletLossZ = self.kvae.CalculateTripletLossZFromState(statesZ,alpha_plot)
        
        self.assertTrue(tripletLossZ == 0)     
        
        return
    
    def CheckFunctionFindConnectedClusterInBatch(self):
        
        transitionMat = np.array([[0.95, 0.05, 0, 0],[0, 1, 0, 0],[0, 0.1, 0.9, 0], [0.1, 0.1, 0, 0.8]])
        transitionMat = torch.as_tensor(transitionMat)
        clustersInBatch = np.array([0,1,3])
        clustersInBatch = torch.as_tensor(clustersInBatch)
        
        currentAssignment = 0
        connectedClustersInBatch = KalmanVariationalAutoencoder.FindConnectedClustersInBatch(transitionMat, clustersInBatch, currentAssignment)
        connectedClustersInBatchCorrect = np.array([1])
        self.assertTrue((connectedClustersInBatch == connectedClustersInBatchCorrect).all())  
        
        currentAssignment = 1
        connectedClustersInBatch = KalmanVariationalAutoencoder.FindConnectedClustersInBatch(transitionMat, clustersInBatch, currentAssignment)
        connectedClustersInBatchCorrect = np.array([])
        self.assertTrue((connectedClustersInBatch == connectedClustersInBatchCorrect).all())  
        
        currentAssignment = 3
        connectedClustersInBatch = KalmanVariationalAutoencoder.FindConnectedClustersInBatch(transitionMat, clustersInBatch, currentAssignment)
        connectedClustersInBatchCorrect = np.array([0,1])
        self.assertTrue((connectedClustersInBatch == connectedClustersInBatchCorrect).all())          
        return
    
    def CheckFunctionFindUnconnectedClusterInBatch(self):
        
        transitionMat = np.array([[0.95, 0.05, 0, 0],[0, 1, 0, 0],[0, 0.1, 0.9, 0], [0.1, 0.1, 0, 0.8]])
        transitionMat = torch.as_tensor(transitionMat)
        clustersInBatch = np.array([0,1,3])
        clustersInBatch = torch.as_tensor(clustersInBatch)
        
        currentAssignment = 0
        connectedClustersInBatch = KalmanVariationalAutoencoder.FindUnconnectedClustersInBatch(transitionMat, clustersInBatch, currentAssignment)
        connectedClustersInBatchCorrect = np.array([3])
        self.assertTrue((connectedClustersInBatch == connectedClustersInBatchCorrect).all())  
        
        currentAssignment = 1
        connectedClustersInBatch = KalmanVariationalAutoencoder.FindUnconnectedClustersInBatch(transitionMat, clustersInBatch, currentAssignment)
        connectedClustersInBatchCorrect = np.array([0,3])
        self.assertTrue((connectedClustersInBatch == connectedClustersInBatchCorrect).all())  
        
        currentAssignment = 3
        connectedClustersInBatch = KalmanVariationalAutoencoder.FindUnconnectedClustersInBatch(transitionMat, clustersInBatch, currentAssignment)
        connectedClustersInBatchCorrect = np.array([])
        self.assertTrue((connectedClustersInBatch == connectedClustersInBatchCorrect).all())          
        return
    
    def CheckFunctionSelectPointInBatchBelongingToClustersSubset(self):
        
        numberOfTestLoops = 100 # how many times to repeat the test
        
        clustersSubset = np.array([0,1])
        clusterAssignments = np.array([[0,0,0,0,1,1,2], [2,1,2,2,2,2,2]])
        clusterAssignments = torch.as_tensor(clusterAssignments)
        statesZ = torch.rand(2,7,4)
        
        # Manually select the state values of the clustersSubset
        elementsCluster0 = statesZ[0:1,0:4,:]
        elementsCluster1 = torch.cat((statesZ[0:1,4:6,:], statesZ[1:2,1:2,:]), dim = 1)
        elementsClusters0and1 = torch.cat((elementsCluster0,elementsCluster1), dim=1)
        
        # Vector to keep which clusters where selected
        selectedClusters = np.zeros(numberOfTestLoops)
        
        for i in range(numberOfTestLoops):
            # run FUNCTION TO TEST
            selectedPoint = KalmanVariationalAutoencoder.SelectPointInBatchBelongingToClustersSubset(clustersSubset, clusterAssignments, statesZ)
            # Check that the point is among the ones it should be from
            self.assertTrue((selectedPoint == elementsClusters0and1).all(2).any())   # <-- CHECK 1
            # Now for checking that not always the same cluster is taken
            # Cluster 0?
            isItCluster0 = (selectedPoint == elementsCluster0).all(2).any()
            isItCluster1 = (selectedPoint == elementsCluster1).all(2).any()
            if isItCluster0:
                selectedClusters[i] = 0
            if isItCluster1:
                selectedClusters[i] = 1
                
        # Check that not always the same cluster was taken
        self.assertFalse((selectedClusters == np.zeros(numberOfTestLoops)).all())  # <-- CHECK 2         
        return
    
    @staticmethod
    def PerformAllTests():
        
        TestKVAE = TestsKVAE()
        TestKVAE.CheckThatTripletLossZIsZeroWhenAllDataBelongsToTheSameCluster()
        TestKVAE.CheckFunctionFindConnectedClusterInBatch()
        TestKVAE.CheckFunctionFindUnconnectedClusterInBatch()
        TestKVAE.CheckFunctionSelectPointInBatchBelongingToClustersSubset()
        print('All tests have been successfully performed')
        return
    
def main():
    TestsKVAE.PerformAllTests()
    
main() 
    
               