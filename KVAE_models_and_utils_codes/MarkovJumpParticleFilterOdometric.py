# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:22:38 2021

@author: giulia
"""

import torch
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

class MarkovJumpParticleFilterOdometric(MJPF.MarkovJumpParticleFilter):
    
    ###########################################################################
    # INITIALIZATION
    
    # Function for re-setting to zero all the variables that are initialized at
    # the beginning of the MJPF running.
    # These are all the variables that depend on the number of used particles.
    # INPUTS:
    # - probabilityOfClusters: probability of being in each cluster.
    def RestartVariables(self, probabilityOfClusters = None):
        
        super(self.__class__, self).RestartVariables(probabilityOfClusters) 
        
        # Velocity
        self.U_predictions = torch.zeros(self.dimensionOfState, self.numberOfParticles).to(device)
        # To keep the 'velocity'
        self.differenceOfMeansBetweenUpdates = torch.zeros(self.dimensionOfState, self.numberOfParticles).to(device)
        # keeping track of cluster movements
        self.passagesBetweenClusters = np.zeros((self.numberOfParticles, 2))

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
    # - z_meaning: what the z state represents
    # - nodesCovPred: covariance of the prediction performed by the D and E matrices.
    def __init__(self, numberOfParticles, dimensionOfState, nodesMean, nodesCov, nodesCovPred, transitionMat, \
                 z_meaning, observationCovariance, resamplingThreshold = 2, transMatsTime = None, 
                 transitionMatExplorationRatio = 0, vicinityTransitionMat = None, \
                 maxClustersTime = None, probabilityOfClusters = None, observationTrustFactor = 1):
        
        super(self.__class__, self).__init__(numberOfParticles, dimensionOfState, nodesMean, nodesCov, transitionMat, 
                 observationCovariance, resamplingThreshold, transMatsTime, transitionMatExplorationRatio,
                 vicinityTransitionMat, maxClustersTime, probabilityOfClusters, observationTrustFactor)

        self.z_meaning               = z_meaning        
        self.nodesCovPred            = nodesCovPred
        
        if self.z_meaning == 0: # pos+ vel            
            self.stateTransitionMat      = torch.eye(self.dimensionOfState).to(device)
            self.stateTransitionMat[2,2] = 0
            self.stateTransitionMat[3,3] = 0
            self.controlMat              = torch.eye(self.dimensionOfState).to(device)

        elif self.z_meaning == 2: #theta + pos + velNorm            
            self.stateTransitionMat      = torch.eye(self.dimensionOfState).to(device)
            self.stateTransitionMat[0,0] = 0
            self.stateTransitionMat[3,3] = 0
            self.controlMat              = torch.eye(self.dimensionOfState).to(device)
            
        # Pointer to the function for performing prediction
        if self.z_meaning == 0: # pos+ vel            
            self.PredictParticle         = self.PredictParticleAtGivenIndexUsingPosVel
        
        elif self.z_meaning == 2: #theta + pos + velNorm            
            self.PredictParticle         = self.PredictParticleAtGivenIndexUsingThetaAndVelNorm
             
        return
   
    ###########################################################################
    # PREDICTION at state level
    
    # Function to predict next state of a particle using a position/velocity model.
    # INPUTS:
    # - particleIndex: index of the particle to consider.
    # OUTPUTS:
    # - predictionMean: predicted mean value
    # - predictionCov: predicted covariance value
    def PredictParticleAtGivenIndexUsingPosVel(self, particleIndex):
        
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
        Q = self.nodesCovPred[int(currentCluster)]
        # KF PREDICTION
        updatedMean = torch.unsqueeze(self.particlesMeans[:, particleIndex], 1)
        updatedCov  = self.particlesCovariances[:,:,particleIndex]
        
        predictionMean, predictionCov = KF.kf_predict(updatedMean.double(), updatedCov.double(),
                                                      self.stateTransitionMat.double(), Q.double(),
                                                      self.controlMat.double(),         U.double())       
        return predictionMean, predictionCov
    
    # Function to predict next state of a particle using a rotational model.
    # INPUTS:
    # - particleIndex: index of the particle to consider.
    # OUTPUTS:
    # - predictionMean: predicted mean value
    # - predictionCov: predicted covariance value
    def PredictParticleAtGivenIndexUsingThetaAndVelNorm(self, particleIndex):
        
        # MEAN
        currentCluster      = self.clusterAssignments[int(particleIndex)]        
        currentTheta        = self.nodesMean[int(currentCluster), 0]
        currentMeanNormVel  = self.nodesMean[int(currentCluster), 3]        
        currentVelocity     = self.differenceOfMeansBetweenUpdates[[1, 2], particleIndex]        
        # Rotation matrix definition
        rotationMatrix      = torch.zeros(2,2).to(device)
        rotationMatrix[0,0] = torch.cos(currentTheta)
        rotationMatrix[0,1] = - torch.sin(currentTheta)
        rotationMatrix[1,0] = torch.sin(currentTheta)
        rotationMatrix[1,1] = torch.cos(currentTheta)        
        U = torch.matmul(rotationMatrix,currentVelocity)        
        U_norm     = torch.sqrt(torch.pow(U[0], 2) + torch.pow(U[1], 2))
        if U_norm != 0:
            U_norm = U/U_norm
            U      = U_norm*currentMeanNormVel
        else:
            U_norm = U
        U_pos = torch.unsqueeze(U, 1)        
        U = torch.zeros(4, 1).to(device)
        U[0, 0] = currentTheta
        U[1, 0] = U_pos[0, 0]
        U[2, 0] = U_pos[1, 0]
        U[3, 0] = currentMeanNormVel       
        self.U_predictions[:, particleIndex] = torch.squeeze(U.clone())       
        # COVARIANCE        
        Q = self.nodesCovPred[int(currentCluster)]        
        # KF PREDICTION      
        updatedMean = torch.unsqueeze(self.particlesMeans[:, particleIndex], 1)
        updatedCov  = self.particlesCovariances[:,:,particleIndex]        
        predictionMean, predictionCov = KF.kf_predict(updatedMean.double(), updatedCov.double(),
                                                      self.stateTransitionMat.double(), Q.double(),
                                                      self.controlMat.double(),         U.double())             
        return predictionMean, predictionCov
    
    ###########################################################################
    # RESAMPLING
    
    # Function to perform resampling.
    def ResampleParticles(self):
        
        newIndicesForSwapping = super(MarkovJumpParticleFilterOdometric, self).ResampleParticles() 
        
        # Perform swapping       
        # First create temporary vectors by copying from the ones already present
        U_predictionsTemp                   = self.U_predictions.clone()
        differenceOfMeansBetweenUpdatesTemp = self.differenceOfMeansBetweenUpdates.clone()
        
        # Change using the temporary vector (so to not generate problems)
        for particleIndex in range(self.numberOfParticles):  
            
            currentSwapIndex = newIndicesForSwapping[particleIndex]

            U_predictionsTemp[:, particleIndex]                   = self.U_predictions[:, currentSwapIndex]
            differenceOfMeansBetweenUpdatesTemp[:, particleIndex] = self.differenceOfMeansBetweenUpdates[:, currentSwapIndex]
            
        # Then save the final results from the temporary vector
        self.U_predictions                   = U_predictionsTemp.clone()
        self.differenceOfMeansBetweenUpdates = differenceOfMeansBetweenUpdatesTemp.clone()
            
        # Weights of the particles
        # Give same weigth to all the particles
        self.particlesWeights      = torch.ones(self.numberOfParticles)/self.numberOfParticles
        
        del U_predictionsTemp, differenceOfMeansBetweenUpdatesTemp
        
        if device.type == "cuda":
            torch.cuda.empty_cache()       
            
        return newIndicesForSwapping
    
    
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
    def FindClosestParticleOfSecondMJPF(self, particleIndex, meansOfOtherMJPF):
        
        currentParticleOfSelfMJPF                                = self.particlesMeans[:,particleIndex]
        indexOfClosestParticle, closestParticle, distanceValue = \
           MJPF.MarkovJumpParticleFilter.find_nearest(meansOfOtherMJPF, currentParticleOfSelfMJPF)
        
        return int(indexOfClosestParticle), closestParticle, distanceValue
    
    # Input:
    # - the mean values of position for the other MJPF.
    #   shape: (stateDim*numParticles), e.g, (4*100)
    # Outputs:
    # - the assignments 
    # - the mean values of the other MJPF, assigned
    # - the distance values
    def FindClosestParticlesOfSecondMJPF(self, meansOfOtherMJPF):
        
        indicesAssignments        = np.zeros(self.numberOfParticles).astype(int)
        #inverseIndicesAssignments = np.zeros(self.numberOfParticles).astype(int)
        meanValuesAssigned        = torch.zeros(meansOfOtherMJPF.shape[0], self.numberOfParticles).to(device)
        distanceValues            = np.zeros(self.numberOfParticles)
        
        # Looping over the number of particles
        for particleIndex in range(self.numberOfParticles):          
            
            indexOfClosestParticle, closestParticle, distanceValue = \
               self.FindClosestParticleOfSecondMJPF(particleIndex, meansOfOtherMJPF)
            
            indicesAssignments[particleIndex]    = int(indexOfClosestParticle)
            #inverseIndicesAssignments[indexOfClosestParticle]    = int(particleIndex)
            meanValuesAssigned[:, particleIndex] = closestParticle
            distanceValues[particleIndex]        = distanceValue
        
        return indicesAssignments, meanValuesAssigned, distanceValues
    
    # Input:
    # - the index of the particles in the current MJPF.
    # - the mean values of position for the other MJPF.
    #   shape: (stateDim*numParticles), e.g, (4*100)
    # Outputs:
    # - the assignments 
    # - the mean values of the other MJPF, assigned
    # - the distance value
    def FindClosestParticleOfSecondMJPFGivenCluster(self, particleIndex, meansOfOtherMJPF, 
                                                    closestCluster, clusterAssignmentsOfOtherMJPF):
        
        # First find which particles belong to the closest cluster
        indicesWhereClosestClusterIs     = np.where(clusterAssignmentsOfOtherMJPF == closestCluster)
        indicesWhereClosestClusterIs     = indicesWhereClosestClusterIs[0]
        # Now select values of other MJPF at those indices
        meansOfOtherMJPFAtClosestCluster = torch.squeeze(meansOfOtherMJPF[:, indicesWhereClosestClusterIs])
        
        currentParticleOfSelfMJPF                                = self.particlesMeans[:,particleIndex]
        
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
    def FindClosestParticlesOfSecondMJPFOfClosestCluster(self, meansOfOtherMJPF, clusterAssignmentsOfOtherMJPF):
        
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
               self.FindClosestParticleOfSecondMJPFGivenCluster(particleIndex, meansOfOtherMJPF, closestCluster,
                                                    clusterAssignmentsOfOtherMJPF)
            
            indicesAssignments[particleIndex]    = int(indexOfClosestParticle)
            #inverseIndicesAssignments[indexOfClosestParticle]    = int(particleIndex)
            meanValuesAssigned[:, particleIndex] = closestParticle
            distanceValues[particleIndex]        = distanceValue
        
        return indicesAssignments, meanValuesAssigned, distanceValues
        
        
    
        