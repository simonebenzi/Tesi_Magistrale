# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:22:38 2021

@author: giulia
"""

import torch
import torch.nn.functional as F
import numpy as np

from KVAE_models_and_utils_codes import KF_torch   as KF
from ConfigurationFiles          import Config_GPU as ConfigGPU 

from KVAE_models_and_utils_codes import MarkovJumpParticleFilter as MJPF

###############################################################################
# ----------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)

###############################################################################
# ------------------    Markov Jump Particle Filter  --------------------------
###############################################################################

class MarkovJumpParticleFilterVideo(MJPF.MarkovJumpParticleFilter):
    
    ###########################################################################
    # INITIALIZATION
    
    # Function for re-setting to zero all the variables that are initialized at
    # the beginning of the MJPF running.
    # These are all the variables that depend on the number of used particles.
    # INPUTS:
    # - probabilityOfClusters: probability of being in each cluster.
    def RestartVariables(self, probabilityOfClusters = None):
        
        super(self.__class__, self).RestartVariables(probabilityOfClusters) 
        
        if self.dimensionOfOdometricState != None:
            # From D matrices
            self.allPredictedParams = torch.zeros(self.dimensionOfOdometricState, self.numberOfParticles).to(device)
        
        return
    
    # Initialization of the MJPF.
    # INPUTS:
    # - numberOfParticles: number of particles of the MJPF
    # - nodesMean: mean of the clusters: (number of clusters, state dimension)
    # - nodesCov: covariance of clusters: (number of clusters, state dimension, state dimension)
    # - transitionMat: transition matrix: (number of clusters, state dimension)
    # - observationCovariance: covariance matrix of the observation
    # - resamplingThreshold: Threshold on the number of effective samples 
    #      for doing resampling. Resampling is performed when we are below the
    #      threshold.
    # - transMatsTime: temporal transition matrices.
    # - vicinityTransitionMat: vicinity transition matrix.
    # - transitionMatExplorationRatio: how much to use the original transition 
    #      matrix and how much to use the additional exploration.
    # - maxClustersTime: maximum time spent in each cluster.
    # - probabilityOfClusters: probability of being in each cluster.
    # INPUTS NOT PRESENT IN BASE OBJECT:
    # - kf: kalman filter of KVAE
    
    def __init__(self, numberOfParticles, dimensionOfState, nodesMean, nodesCov, transitionMat, \
                 observationCovariance, kf, resamplingThreshold = 2, 
                 transMatsTime = None, vicinityTransitionMat = None, transitionMatExplorationRatio = 0,  
                 maxClustersTime = None, probabilityOfClusters = None, observationTrustFactor = 1):

        self.dimensionOfOdometricState   = kf.clusteringDimension
        
        super(self.__class__, self).__init__(numberOfParticles, dimensionOfState, nodesMean, nodesCov, transitionMat, 
                 observationCovariance, resamplingThreshold, transMatsTime, vicinityTransitionMat, transitionMatExplorationRatio,
                 maxClustersTime, probabilityOfClusters, observationTrustFactor)
        
        self.PredictParticle             = self.PredictParticleAtGivenIndexUsingAMatrix
        self.kf                          = kf
          
        return
    
    ###########################################################################
    # FUNCTIONS TO COMPARE TWO MJPFs
    
    # A function to compare two MJPFs and see if they are the same.
    # INPUTS:
    # - otherMJPF: the other MJPF against which to compare the current one.
    def CompareMJPFs(self, otherMJPF):
        
        MJPFsAreEqual     = super(self.__class__, self).CompareMJPFs(otherMJPF) 
        
        if self.dimensionOfOdometricState != None:
            allPredictedParamsAreEqual = (self.allPredictedParams==otherMJPF.allPredictedParams).all()
        
        MJPFsAreEqual     = MJPFsAreEqual and allPredictedParamsAreEqual
        
        return MJPFsAreEqual
   
    ###########################################################################
    # PREDICTION at state level
    
    # Function to predict next state of a particle using the KVAE A matrices model.
    # INPUTS:
    # - particleIndex: index of the particle to consider.
    # OUTPUTS:
    # - predictionMean: predicted mean value
    # - predictionCov: predicted covariance value
    def PredictParticleAtGivenIndexUsingAMatrix(self, particleIndex):

        # MEAN
        currentCluster = self.clusterAssignments[int(particleIndex)]
        currentVelX    = self.nodesMean[int(currentCluster), 2]
        currentVelY    = self.nodesMean[int(currentCluster), 3]
        U = torch.zeros(4, 1).to(device)
        U[0, 0] = currentVelX
        U[1, 0] = currentVelY
        U[2, 0] = currentVelX
        U[3, 0] = currentVelY        
        self.U_predictions[:, particleIndex] = torch.squeeze(U.clone())        
        # COVARIANCE
        Q = self.nodesCov[int(currentCluster),:,:]        
        # KF PREDICTION
        updatedMean = torch.unsqueeze(self.particlesMeans[:, particleIndex], 1)
        updatedCov  = self.particlesCovariances[:,:,particleIndex]
        
        predictionMean, predictionCov = KF.kf_predict(updatedMean.double(), updatedCov.double(),
                                                      self.stateTransitionMat.double(), Q.double(),
                                                      self.controlMat.double(),         U.double())        
        return predictionMean, predictionCov
    
    ###########################################################################
    # Function to predict the odometry values given D and E matrices
    
    # Function to predict the odometry values of each particle, using the 
    # D and E matrices provided by the kf variable.
    # OUTPUTS:
    # - allPredictedParams: predicted odometry values 
    def FindParticlesUsingDMatrices(self):
        
        D_matrices = self.kf.D
        E_matrices = self.kf.E
        nodesMean  = self.kf.nodesMean 
        
        for particleIndex in range(self.numberOfParticles): 
            
            currentCluster  = int(self.clusterAssignments[particleIndex])
            
            currentState    = self.particlesMeans[:,particleIndex]
            currentD        = D_matrices[currentCluster, :,:]
            currentE        = E_matrices[currentCluster, :]
            currentNodeMean = nodesMean[currentCluster, :]
            
            predictedParams = torch.matmul(currentD, currentState) + currentE + currentNodeMean
            
            self.allPredictedParams[:,particleIndex] = predictedParams.clone()
        
        return self.allPredictedParams
    
    ###########################################################################
    # UPDATE at state level
    
    # Function to update the state of a particle using the KVAE C matrices model.
    # INPUTS:
    # - a_mu: observations of video state 'a'.
    def PerformUpdateOfParticlesUsingMatrices(self, a_mu):
        
        # 4) Perform video UPDATE using video clustering
        # First we perform the update phase of the KVAE for getting the updated 
        # video value using the alpha that we obtain from video clustering (and
        # from past sequencing)
        # Note that we have MULTIPLE UPDATES, one for each particle!
        for particleIndex in range(self.numberOfParticles): 
            
            # Cluster of the current particle
            current_cluster = self.clusterAssignments[particleIndex]
            # A, B, C, D, E, and nodeMean matrix for the current cluster
            A,B,C           = self.kf.extract_A_B_C_of_cluster_index(current_cluster)
        
            mu_t, Sigma_t   = self.kf.perform_update_given_C(torch.unsqueeze(self.particlesCovariances[:,:,particleIndex], 0), 
                                                             torch.unsqueeze(self.particlesMeans[:,particleIndex], 0), 
                                                             a_mu, C)
            
            self.particlesMeans[:, particleIndex]        = torch.squeeze(mu_t)
            self.particlesCovariances[:,:,particleIndex] = torch.squeeze(Sigma_t)
            
        self.particlesMeansUpdated       = self.particlesMeans.clone()
        self.particlesCovariancesUpdated = self.particlesCovariances.clone()
            
        return
    
    ###########################################################################
    # PREDICTION at state level
    
    # Function to predict next state of a particle using the KVAE A model
    def PerformPredictionOfParticlesUsingMatrices(self):
        
        for particleIndex in range(self.numberOfParticles):
            
            # Cluster of the current particle
            current_cluster     = self.clusterAssignments[particleIndex]
            # A, B, C, D, E, and nodeMean matrix for the current cluster
            A,B,C               = self.kf.extract_A_B_C_of_cluster_index(current_cluster)
            
            
            mu_pred, Sigma_pred = self.kf.perform_prediction_given_A_B(torch.unsqueeze(self.particlesMeans[:,particleIndex], 0), 
                                                                       torch.unsqueeze(self.particlesCovariances[:,:,particleIndex], 0),
                                                                       A, B)

            self.particlesMeans[:, particleIndex]        = torch.squeeze(mu_pred.clone())
            self.particlesCovariances[:,:,particleIndex] = torch.squeeze(Sigma_pred.clone())
            
        self.particlesMeansPredicted       = self.particlesMeans.clone()
        self.particlesCovariancesPredicted = self.particlesCovariances.clone()
            
        return
    
    ###########################################################################
    # For case with other MJPF combined
    
    # Input:
    # - the index of the particles in the current MJPF.
    # - the mean values of position for the other MJPF.
    #   shape: (stateDim*numParticles), e.g, (4*100)
    # Outputs:
    # - the assignments 
    # - the mean values of the other MJPF, assigned
    # - the distance value
    def FindClosestParticleOfSecondMJPF(self, particleIndex, updatedValuesFromDMatrices, meansOfOtherMJPF):
        
        currentParticleOfSelfMJPF                                = updatedValuesFromDMatrices[:,particleIndex]
        indexOfClosestParticle, closestParticle, distanceValue = \
           MJPF.MarkovJumpParticleFilter.find_nearest(meansOfOtherMJPF, currentParticleOfSelfMJPF)
        
        return int(indexOfClosestParticle), closestParticle, distanceValue
    
    # Function to find the closest particle of a second MJPF given the cluster.
    def FindClosestParticleOfSecondMJPFGivenCluster(self, particleIndex, updatedValuesFromDMatrices, 
                                                    meansOfOtherMJPF, closestCluster, 
                                                    clusterAssignmentsOfOtherMJPF):
        
        # First find which particles belong to the closest cluster
        indicesWhereClosestClusterIs     = np.where(clusterAssignmentsOfOtherMJPF == closestCluster)
        indicesWhereClosestClusterIs     = indicesWhereClosestClusterIs[0]
        # Now select values of other MJPF at those indices
        meansOfOtherMJPFAtClosestCluster = meansOfOtherMJPF[:, indicesWhereClosestClusterIs]
        
        # Find closest point
        currentParticleOfSelfMJPF                                = updatedValuesFromDMatrices[:,particleIndex]
        indexOfClosestParticle, closestParticle, distanceValue = \
           MJPF.MarkovJumpParticleFilter.find_nearest(meansOfOtherMJPFAtClosestCluster, currentParticleOfSelfMJPF)
           
        # Bring back the index to original scale
        indexFinal = int(indicesWhereClosestClusterIs[int(indexOfClosestParticle)])
        
        return indexFinal, closestParticle, distanceValue
    
    # Input:
    # - the mean values of position for the other MJPF.
    #   shape: (stateDim*numParticles), e.g, (4*100)
    # Outputs:
    # - the assignments 
    # - the mean values of the other MJPF, assigned
    # - the distance values
    def FindClosestParticlesOfSecondMJPF(self, updatedValuesFromDMatrices, meansOfOtherMJPF):
        
        indicesAssignments        = np.zeros(self.numberOfParticles).astype(int)
        #inverseIndicesAssignments = np.zeros(self.numberOfParticles).astype(int)
        meanValuesAssigned        = torch.zeros(meansOfOtherMJPF.shape[0], self.numberOfParticles).to(device)
        distanceValues            = np.zeros(self.numberOfParticles)
        
        # Looping over the number of particles
        for particleIndex in range(self.numberOfParticles):          
            
            indexOfClosestParticle, closestParticle, distanceValue = \
               self.FindClosestParticleOfSecondMJPF(particleIndex, updatedValuesFromDMatrices, meansOfOtherMJPF)
            
            indicesAssignments[particleIndex]    = int(indexOfClosestParticle)
            #inverseIndicesAssignments[indexOfClosestParticle]    = int(particleIndex)
            meanValuesAssigned[:, particleIndex] = closestParticle
            distanceValues[particleIndex]        = distanceValue
        
        return indicesAssignments, meanValuesAssigned, distanceValues
    
    # This function does not directly assign the closest particle of the other 
    # MJPF, conversely to what the function 'FindClosestParticlesOfSecondMJPF' did,
    # but first considers which is the closest cluster assignment of the other
    # MJPF w.r.t. the cluster assignment of our MJPF's particles, for each particle.
    # In this way, at intersections, odometry particles will go in different
    # directions instead of being all reconduced to a single video particle, which
    # can easily happen.
    def FindClosestParticlesOfSecondMJPFOfClosestCluster(self, updatedValuesFromDMatrices, meansOfOtherMJPF, 
                                                         clusterAssignmentsOfOtherMJPF):
        
        indicesAssignments        = np.zeros(self.numberOfParticles).astype(int)
        #inverseIndicesAssignments = np.zeros(self.numberOfParticles).astype(int)
        meanValuesAssigned        = torch.zeros(meansOfOtherMJPF.shape[0], self.numberOfParticles).to(device)
        distanceValues            = np.zeros(self.numberOfParticles)
        
        # Looping over the number of particles
        for particleIndex in range(self.numberOfParticles):   
            
            # First the closest cluster of other MJPF to clusters of this MJPF
            closestCluster = self.FindClosestClusterOfSecondMJPFSingleParticle(
                clusterAssignmentsOfOtherMJPF, particleIndex)
            
            indexOfClosestParticle, closestParticle, distanceValue = \
               self.FindClosestParticleOfSecondMJPFGivenCluster(particleIndex, updatedValuesFromDMatrices, 
                                                                meansOfOtherMJPF, closestCluster,
                                                                clusterAssignmentsOfOtherMJPF)
            
            indicesAssignments[particleIndex]    = int(indexOfClosestParticle)
            #inverseIndicesAssignments[indexOfClosestParticle]    = int(particleIndex)
            meanValuesAssigned[:, particleIndex] = closestParticle
            distanceValues[particleIndex]        = distanceValue
        
        return indicesAssignments, meanValuesAssigned, distanceValues
    
    ###########################################################################
    # KVAE-specific functions
    
    # Anomalies calculations between predicted 'a' state and observed 'a' state
    # for each particle.
    # INPUTS:
    # - realAState: observed 'a' state value.
    # OUTPUTS:
    # - anomalies
    def FindAStatesPredictionVsObservationAnomalies(self, realAState):
        
        #if(realAState.ndim) == 1:
        #    realAState = torch.squeeze(realAState, 0)
        
        # Initialize anomaly vector
        anomalies = torch.zeros(self.numberOfParticles).to(device)
        
        # Looping over the number of particles
        for particleIndex in range(self.numberOfParticles):  
            
            currentParticleAPrediction = self.ObtainAPredictionFromZPredictionGivenParticleIndex(particleIndex)
            
            # 3) Calculate anomaly
            currentAnomaly             = F.mse_loss(currentParticleAPrediction, 
                                                    realAState,size_average=False)   
            
            anomalies[particleIndex] = currentAnomaly

        return anomalies
    
    # Function to find the 'a' state value from the 'z' state value of the 
    # prediction for each particle.
    # INPUTS:
    # - particleIndex: index of particle.
    # OUTPUTS:
    # - currentParticleAPrediction: value of 'a' state predicted for current particle.
    def ObtainAPredictionFromZPredictionGivenParticleIndex(self, particleIndex):
        
        # 1) Get the performed prediction on z
        currentParticleZPrediction = self.particlesMeansPredicted[:, particleIndex]
        currentParticleZPrediction = torch.unsqueeze(currentParticleZPrediction, 0)
        # 2) Pass from prediction on z to prediction on a
        # Cluster of the current particle
        current_cluster = self.clusterAssignments[particleIndex]
        currentParticleAPrediction = self.kf.ObtainAValueFromZValueGivenCluster(
            currentParticleZPrediction, current_cluster)
        
        return currentParticleAPrediction

    # Function to find the 'a' state value from the 'z' state value of the 
    # prediction for all particles.
    def ObtainAPredictionFromZPredictionForAllParticles(self):
        
        # Initialize vector of predicted states
        self.aStatesPredictions = torch.zeros(self.dimensionOfObservation, self.numberOfParticles)
        
        # Looping over the number of particles
        for particleIndex in range(self.numberOfParticles): 
            # Find a state predictions
            self.aStatesPredictions[:,particleIndex] = self.ObtainAPredictionFromZPredictionGivenParticleIndex(particleIndex)
        
        return
    
    # Function to find the 'a' state value from the 'z' state value of the 
    # update for each particle.
    # INPUTS:
    # - particleIndex: index of particle.
    # OUTPUTS:
    # - currentParticleAPrediction: value of 'a' state updated for current particle.
    def ObtainAUpdateFromZUpdateGivenParticleIndex(self, particleIndex):
        
        # 1) Get the performed prediction on z
        currentParticleZUpdate = self.particlesMeansUpdated[:, particleIndex]
        currentParticleZUpdate = torch.unsqueeze(currentParticleZUpdate, 0)
        # 2) Pass from update on z to update on a
        # Cluster of the current particle
        current_cluster = self.clusterAssignments[particleIndex]
        currentParticleAUpdate = self.kf.ObtainAValueFromZValueGivenCluster(
            currentParticleZUpdate, current_cluster)
        
        return currentParticleAUpdate
    
    # Function to find the 'a' state value from the 'z' state value of the 
    # update for all particles.
    def ObtainAUpdateFromZUpdateForAllParticles(self):
        
        # Initialize vector of updated states
        self.aStatesUpdates = torch.zeros(self.dimensionOfObservation, self.numberOfParticles)
        
        # Looping over the number of particles
        for particleIndex in range(self.numberOfParticles): 
            # Find a state updates
            self.aStatesUpdates[:,particleIndex] = self.ObtainAUpdateFromZUpdateGivenParticleIndex(particleIndex)
        
        return
    
    # Function to find the 'a' state value from the 'z' state value of the 
    # update and prediction for all particles.
    def ObtainAPredictionAndUpdateFromZPredictionAndUpdateForAllParticles(self):
        
        self.ObtainAPredictionFromZPredictionForAllParticles()
        self.ObtainAUpdateFromZUpdateForAllParticles()
        
        return