# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:22:38 2021

@author: giulia
"""

# Class for the Markov Jump Particle Filter.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
import random

from KVAE_models_and_utils_codes import KF_torch   as KF
from KVAE_models_and_utils_codes import Distance_utils as d_utils
from ConfigurationFiles          import Config_GPU as ConfigGPU 

###############################################################################
# ----------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)

np.random.seed(int(time.time()))

###############################################################################
# ------------------    Markov Jump Particle Filter  --------------------------
###############################################################################

class MarkovJumpParticleFilter(nn.Module):
    
    ###########################################################################
    # INITIALIZATION
    
    # Function for re-setting to zero all the variables that are initialized at
    # the beginning of the MJPF running.
    # These are all the variables that depend on the number of used particles.
    # INPUTS:
    # - probabilityOfClusters: probability of being in each cluster.
    def RestartVariables(self, probabilityOfClusters = None):
    
        # Weights of the particles
        # Start by giving same weigth to all the particles
        self.particlesWeights        = torch.ones(self.numberOfParticles)/self.numberOfParticles
        # This indicates which are the clusters of each particle
        self.clusterAssignments      = np.zeros(self.numberOfParticles)
        # Mean value of particles
        self.particlesMeans          = torch.zeros(self.dimensionOfState, self.numberOfParticles).to(device)
        # Covariance of Particles
        self.particlesCovariances    = torch.zeros(self.dimensionOfState, self.dimensionOfState, self.numberOfParticles).to(device)
        # Time spent in the current cluster, for each particle
        self.timeInClusters          = torch.zeros(self.numberOfParticles).to(device)
        
        # To distinguish between prediction and update
        self.particlesMeansUpdated         = torch.zeros(self.dimensionOfState, self.numberOfParticles).to(device)
        self.particlesCovariancesUpdated   = torch.zeros(self.dimensionOfState, self.dimensionOfState, self.numberOfParticles).to(device)
        self.particlesMeansPredicted       = torch.zeros(self.dimensionOfState, self.numberOfParticles).to(device)
        self.particlesCovariancesPredicted = torch.zeros(self.dimensionOfState, self.dimensionOfState, self.numberOfParticles).to(device)
        
        # This is to actually initialize with meaningful values 
        # clusterAssignments, particlesMeans and particlesCovariances
        # If a probability vector for the cluster is given, we use it (this
        # could be the case of a priori probabilities for each cluster or
        # a given probability).
        # Otherwise we treat all clusters as having equal probability
        if probabilityOfClusters == None:
            self.InitializeParticlesBasedOnGivenClusterProbabilities(torch.ones(self.numberOfClusters)/self.numberOfParticles)
        else:
            self.InitializeParticlesBasedOnGivenClusterProbabilities(probabilityOfClusters)
            
        # keeping track of cluster movements
        self.passagesBetweenClusters     = np.zeros((self.numberOfParticles, 2))
    
        return
    
    # Initialization of the MJPF.
    # INPUTS:
    # - numberOfParticles: number of particles of the MJPF
    # - dimensionOfState: dimension of the state
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
    def __init__(self, numberOfParticles, dimensionOfState, nodesMean, nodesCov, transitionMat, 
                 observationCovariance, resamplingThreshold = 2, transMatsTime = None, vicinityTransitionMat = None, 
                 transitionMatExplorationRatio = 0, maxClustersTime = None, probabilityOfClusters = None, 
                 observationTrustFactor = 1):
        super(MarkovJumpParticleFilter, self).__init__()
        
        # PARTICLE FILTER 
        # How many particles the Particle filter over odometry should have?
        self.numberOfParticles       = numberOfParticles
        # Number of clusters
        self.numberOfClusters        = nodesMean.shape[0]
        # State dimension
        self.dimensionOfState        = dimensionOfState
        self.dimensionOfObservation  = observationCovariance.shape[0]
        
        # Vocabulary
        self.nodesMean               = nodesMean
        self.nodesCov                = nodesCov
        self.transitionMat           = transitionMat
        # Temporal transition matrices
        self.temporalTransitionMatrix= transMatsTime
        self.maxClustersTime         = maxClustersTime
        self.vicinityTransitionMat   = vicinityTransitionMat
        
        self.observationMatrix         = torch.eye(self.dimensionOfState).to(device)
        self.observationCovariance     = observationCovariance.to(device)
        self.observationCovarianceInit = observationCovariance.to(device)
        
        self.transitionMatExplorationRatio = transitionMatExplorationRatio
        self.resamplingThreshold           = resamplingThreshold
        
        self.histogramPredictedParticles   = np.ones(self.numberOfClusters)/self.numberOfClusters
        self.KLDAbnMax                     = 1000
        
        self.observationTrustFactor = observationTrustFactor
        
        # Initializing variables that depend on the number of particles
        self.RestartVariables(probabilityOfClusters)
           
        return
    
    ###########################################################################
    # FUNCTIONS TO COMPARE TWO MJPFs
    
    # A function to compare two MJPFs and see if they are the same.
    # INPUTS:
    # - otherMJPF: the other MJPF against which to compare the current one.
    def CompareMJPFs(self, otherMJPF):
        
        # Compare all the changing attributes of the two MJPFs
        
        nodesMeanAreEqual                     = (self.nodesMean==otherMJPF.nodesMean).all()
        #nodesCovAreEqual                      = (self.nodesCov==otherMJPF.nodesCov).all()
        transitionMatAreEqual                 = (self.transitionMat==otherMJPF.transitionMat).all()
        maxClustersTimeAreEqual               = (self.maxClustersTime==otherMJPF.maxClustersTime).all()
        
        particlesWeightsAreEqual              = (self.particlesWeights==otherMJPF.particlesWeights).all()
        clusterAssignmentsAreEqual            = (self.clusterAssignments==otherMJPF.clusterAssignments).all()
        
        meansAreEqual                         = (self.particlesMeans==otherMJPF.particlesMeans).all()
        covsAreEqual                          = (self.particlesCovariances==otherMJPF.particlesCovariances).all()
        
        particlesMeansUpdatedAreEqual         = (self.particlesMeansUpdated==otherMJPF.particlesMeansUpdated).all()
        particlesCovariancesUpdatedAreEqual   = (self.particlesCovariancesUpdated==otherMJPF.particlesCovariancesUpdated).all()
        
        particlesMeansPredictedAreEqual       = (self.particlesMeansPredicted==otherMJPF.particlesMeansPredicted).all()
        particlesCovariancesPredictedAreEqual = (self.particlesCovariancesPredicted==otherMJPF.particlesCovariancesPredicted).all()
        
        timeInClustersAreEqual                = (self.timeInClusters==otherMJPF.timeInClusters).all()
        
        passagesBetweenClustersAreEqual       = (self.passagesBetweenClusters==otherMJPF.passagesBetweenClusters).all()
        
        # Final verdict
        MJPFsAreEqual = particlesWeightsAreEqual and clusterAssignmentsAreEqual and meansAreEqual and covsAreEqual and \
           particlesMeansUpdatedAreEqual and particlesCovariancesUpdatedAreEqual and \
           particlesMeansPredictedAreEqual and particlesCovariancesPredictedAreEqual and timeInClustersAreEqual and \
           nodesMeanAreEqual and transitionMatAreEqual and maxClustersTimeAreEqual and \
           passagesBetweenClustersAreEqual
                  
        return MJPFsAreEqual
    
    ###########################################################################
    # FUNCTIONS TO INITIALIZE PARTICLES' PROBABILITIES
    
    # Function to assign a mean and covariance value to a particle of the MJPF, 
    # identified by its index.
    # INPUTS:
    # - mean: mean value to assign;
    # - cov: covariance value to assign;
    # - particleIndex: index of the particle.
    def AssignParticleMeanAndCovGivenValue(self, mean, cov, particleIndex):
        
        mean = torch.squeeze(mean)
        cov  = torch.squeeze(cov)

        sampledPoint  = np.random.multivariate_normal(mean = mean.cpu(), cov = cov.cpu())
        sampledPoint  = torch.from_numpy(sampledPoint).to(device)
        
        self.particlesMeans[:, particleIndex]          = sampledPoint.to(device)
        self.particlesCovariances[:, :, particleIndex] = cov.to(device)
        
        self.particlesMeansUpdated[:, particleIndex]          = sampledPoint.to(device)
        self.particlesCovariancesUpdated[:, :, particleIndex] = cov.to(device)
        
        self.particlesMeansPredicted[:, particleIndex]          = sampledPoint.to(device)
        self.particlesCovariancesPredicted[:, :, particleIndex] = cov.to(device)
        
        return
    
    # Allows to initialize all the particles with a mean and covariance.
    # INPUTS:
    # - mean: mean value to assign;
    # - cov: covariance value to assign;
    def InitializeParticlesMeanGivenSingleValue(self, mean, covariance):
        
        for particleIndex in range(self.numberOfParticles):  
            
            self.AssignParticleMeanAndCovGivenValue(mean, covariance, particleIndex)
        
        return
    
    # Assign to particle at index 'particleIndex' a value based on the assigned
    # cluster 'assignedCluster'.
    # This modifies both 'particlesMeans', 'particlesCovariances' and 
    # 'particlesMeansUpdated' and 'particlesCovariancesUpdated'.
    # INPUTS:
    # - particleIndex: index of the particle.
    # - assignedCluster: cluster of which we take mean and covariace to assign to the particle.
    def AssignParticlesMeansAndCovarianceGivenCluster(self, particleIndex, assignedCluster):
        
        meanOfCluster = self.nodesMean[int(assignedCluster), :]
        covOfCluster  = self.nodesCov[int(assignedCluster),:,:].clone()
        
        self.AssignParticleMeanAndCovGivenValue(meanOfCluster, covOfCluster, particleIndex)
        
        return
    
    # This function initializes the particles to a value in the clusters based on a
    # vector assignment 
    # - clusterAssignments: vector containing the cluster assignments
    def InitializeParticlesBasedOnGivenClusterAssignments(self, clusterAssignments):

        for particleIndex in range(self.numberOfParticles):
            
            self.InitializeParticleBasedOnGivenClusterAssignmentsAndIndex(clusterAssignments, particleIndex)
        
        return
    
    # This function initializes the particles to a value in the clusters based on a 
    # given vector of cluster probabilities.
    # This is useful for example to suppose a first probability of odometry location
    # based on some other sensor (e.g., video) which could provide that probability.
    # INPUT:
    # - clusterProbabilities: a vector of as many elements as the number of clusters
    #                         This contains the probability of being in each cluster.
    def InitializeParticlesBasedOnGivenClusterProbabilities(self, clusterProbabilities):
        
        for particleIndex in range(self.numberOfParticles):           
            self.InitializeParticleGivenClusterProbabilities(clusterProbabilities, particleIndex)
        
        return
    
    # This function (re)-initializes inly a percentage of particles based on a 
    # given vector of cluster probabilities.
    # INPUT:
    # - clusterProbabilities: a vector of as many elements as the number of clusters
    #                         This contains the probability of being in each cluster.
    # - percentageOfParticles: percentage of particles to reinitialize.
    def InitializePercentageOfParticlesBasedOnGivenClusterProbabilities(self, clusterProbabilities, percentageOfParticles):
        
        number_of_particles_to_retake     = int(np.floor(percentageOfParticles*self.numberOfParticles))

        for particleIndex in range(number_of_particles_to_retake):           
            self.InitializeParticleGivenClusterProbabilities(clusterProbabilities, particleIndex)
        
        return
    
    # This function initializes a particle at a given index with mean and 
    # covariance of a cluster, picking the cluster using a vector of cluster
    # probabilities.
    # INPUTS:
    # - clusterProbabilities: a vector of as many elements as the number of clusters
    #                         This contains the probability of being in each cluster.
    # - particleIndex: index of the particle.
    def InitializeParticleGivenClusterProbabilities(self, clusterProbabilities, particleIndex):
        
        # Select a random cluster for the particle 
        currentCluster = random.choices(np.arange(self.numberOfClusters), weights = clusterProbabilities, k = 1)[0]

        # Assignment of cluster
        self.clusterAssignments[particleIndex] = currentCluster
        # Assignment 
        self.AssignParticlesMeansAndCovarianceGivenCluster(particleIndex, currentCluster)
        self.timeInClusters[particleIndex] = 0
        #self.U_predictions
        #self.differenceOfMeansBetweenUpdates
        
        return
    
    # Assign to 'updated' value of particle at index 'particleIndex' a value 
    # based on the assigned cluster 'assignedCluster'.
    # This only modifyes the 'updated' mean and cov but not 'particlesMeans' 
    # 'particlesCovariances'.
    # INPUTS:
    # - particleIndex: index of the particle.
    # - assignedCluster: cluster of which we take mean and covariace to assign to the particle.
    def InitializeParticleBasedOnGivenClusterAssignmentAndIndex(self,currentCluster, particleIndex):
        
        # Assignment
        self.AssignParticlesMeansAndCovarianceGivenCluster(particleIndex, currentCluster)
        self.timeInClusters[particleIndex] = 0
       
        return
    
    # Assign to 'updated' value of particle at index 'particleIndex' a value 
    # based on its assigned cluster inside 'clusterAssignments'.
    # This only modifyes the 'updated' mean and cov but not 'particlesMeans' 
    # 'particlesCovariances'.
    # It calls 'InitializeParticleBasedOnGivenClusterAssignmentsAndIndex'.
    # INPUTS:
    # - particleIndex: index of the particle.
    # - clusterAssignments: a vector containing the cluster assignments of all
    #   the particles.
    def InitializeParticleBasedOnGivenClusterAssignmentsAndIndex(self, clusterAssignments, particleIndex):
        
        # Select a random cluster for the particle 
        currentCluster = clusterAssignments[particleIndex]
        # Assignment of cluster
        self.clusterAssignments[particleIndex] = currentCluster        
        self.InitializeParticleBasedOnGivenClusterAssignmentAndIndex(currentCluster, particleIndex)
        
        return
    
    ###########################################################################
    # Finding indices of particles with lower weights
    
    # Select the indices of the 'numberOfParticlesToSelect' with smaller weights.
    # INPUTS:
    # - numberOfParticlesToSelect: how many indices to select
    def SelectIndicesOfParticlesWithLowerWeights(self, numberOfParticlesToSelect):
        
        # In the case a too high number was given
        if numberOfParticlesToSelect > self.numberOfParticles:
            numberOfParticlesToSelect = self.numberOfParticles
        # Sort the weights from smallest to biggest
        sortedParticlesByWeight = np.argsort(self.particlesWeights)
        # Select the indices of the 'numberOfParticlesToSelect' with smaller weights
        selectedIndices = sortedParticlesByWeight[:numberOfParticlesToSelect]
        
        return selectedIndices

    ###########################################################################
    # UPDATE at state level
    
    # Performing update of mean and covariance of a particle identified by its
    # index. The observation covariance used is the one with which the MJPF
    # was initialized.
    # INPUTS:
    # - particleIndex: index of the particle to choose.
    # - observedState: observation.
    def UpdateParticleMeanAndCovarianceAtGivenIndex(self, particleIndex, observedState):
        
        # Predicted mean and covariance are the values in particlesMeans and particlesCovariances
        predictionMean = torch.unsqueeze(self.particlesMeans[:, particleIndex], 1)
        predictionCov  = self.particlesCovariances[:,:,particleIndex]
        # Observation
        observedState  = torch.unsqueeze(observedState, 1)
        
        # UPDATE
        updatedMean, updatedCov = KF.kf_update(predictionMean, predictionCov, observedState, 
                                               self.observationMatrix, self.observationCovariance)
        
        # Save updated mean and covariance in particlesMeans and particlesCovariances
        self.particlesMeans[:, particleIndex]        = torch.squeeze(updatedMean)
        self.particlesCovariances[:,:,particleIndex] = updatedCov
        
        return
    
    # Performing update of mean and covariance of all particles.
    # INPUTS:
    # - observedState: observation.
    def UpdateParticlesMeansAndCovariances(self, observedState):
        
        # Looping over the number of particles
        for particleIndex in range(self.numberOfParticles):            
            self.UpdateParticleMeanAndCovarianceAtGivenIndex(particleIndex, observedState)
            
        self.differenceOfMeansBetweenUpdates = self.particlesMeans - self.particlesMeansUpdated
        
        self.particlesMeansUpdated       = self.particlesMeans.clone()
        self.particlesCovariancesUpdated = self.particlesCovariances.clone()
        
        return
    
    # Performing update of mean and covariance of all particles.
    # Not a single observation is given, but a different observation for each
    # particle.
    # INPUTS:
    # - observedStates: observations.
    def UpdateParticlesMeansAndCovariancesGivenDifferentObservedStates(self, observedStates):
        
        # Looping over the number of particles
        for particleIndex in range(self.numberOfParticles):                       
            self.UpdateParticleMeanAndCovarianceAtGivenIndex(particleIndex, torch.squeeze(observedStates[:,particleIndex]))
            
        self.differenceOfMeansBetweenUpdates = self.particlesMeans.clone() - self.particlesMeansUpdated.clone()
        
        self.particlesMeansUpdated           = self.particlesMeans.clone()
        self.particlesCovariancesUpdated     = self.particlesCovariances.clone()
        
        return
    
    # Performing update of mean and covariance of a particle identified by its
    # index. Observation covariance is also given.
    # INPUTS:
    # - particleIndex: index of the particle to choose.
    # - observedState: observation.
    # - observationCovariance: covariance of observation.
    def UpdateParticleMeanAndCovarianceAtGivenIndexGivenObservationCovariances(self, particleIndex, observedState, observationCovariance):
        
        # Predicted mean and covariance are the values in particlesMeans and particlesCovariances
        predictionMean = torch.unsqueeze(self.particlesMeans[:, particleIndex], 1)
        predictionCov  = self.particlesCovariances[:,:,particleIndex]
        # Observation
        observedState  = torch.unsqueeze(observedState, 1)
        
        # UPDATE
        updatedMean, updatedCov = KF.kf_update(predictionMean.double(), predictionCov.double(), observedState.double(), 
                                               self.observationMatrix.double(), observationCovariance.double())
        
        # Save updated mean and covariance in particlesMeans and particlesCovariances
        self.particlesMeans[:, particleIndex]        = torch.squeeze(updatedMean)
        self.particlesCovariances[:,:,particleIndex] = updatedCov
        
        return
    
    # Performing update of mean and covariance of all particles.
    # Not a single observation is given, but a different observation for each
    # particle.
    # Covariance is defined based on those of another MJPF.
    # INPUTS:
    # - observedStates: observations.
    # - observationMatrices: set of observation covariances, one for each cluster.
    # - clusterAssignments: cluster assignments from which to take the covariances.
    def UpdateParticlesMeansAndCovariancesGivenDifferentObservedStatesAndDifferentObservationCovariances(self,
                                                                                                      observedStates, 
                                                                                                      observationMatrices,
                                                                                                      clusterAssignments):
        # Looping over the number of particles
        for particleIndex in range(self.numberOfParticles):
            
            currentClusterOtherMJPF      = clusterAssignments[particleIndex]
            currentObservationCovariance = observationMatrices[currentClusterOtherMJPF]/self.observationTrustFactor

            self.UpdateParticleMeanAndCovarianceAtGivenIndexGivenObservationCovariances(particleIndex, 
                                                                                   torch.squeeze(observedStates[:,particleIndex]),
                                                                                   currentObservationCovariance)
            
        self.differenceOfMeansBetweenUpdates = self.particlesMeans.clone() - self.particlesMeansUpdated.clone()
        self.particlesMeansUpdated           = self.particlesMeans.clone()
        self.particlesCovariancesUpdated     = self.particlesCovariances.clone()
        
        return 
    
    # Performing update of mean and covariance of all particles.
    # Not a single observation is given, but a different observation for each
    # particle.
    # Covariance is defined based on those of the clusters of the MJPF.
    # INPUTS:
    # - observedStates: observations.
    # - observationMatrices: set of observation covariances, one for each cluster.
    def UpdateParticlesMeansAndCovariancesGivenDifferentObservedStatesAndDifferentObservationCovariancesSingleMJPF(self, 
                                                                                                                   observedStates, 
                                                                                                                   observationMatrices):
        # Looping over the number of particles
        for particleIndex in range(self.numberOfParticles):
            
            currentCluster               = self.clusterAssignments[particleIndex]
            currentObservationCovariance = observationMatrices[int(currentCluster)]/self.observationTrustFactor
            currentObservedState         = torch.squeeze(observedStates[:,particleIndex])

            self.UpdateParticleMeanAndCovarianceAtGivenIndexGivenObservationCovariances(particleIndex, 
                                                                                   currentObservedState,
                                                                                   currentObservationCovariance)
            
        self.differenceOfMeansBetweenUpdates = self.particlesMeans.clone() - self.particlesMeansUpdated.clone()       
        self.particlesMeansUpdated           = self.particlesMeans.clone()
        self.particlesCovariancesUpdated     = self.particlesCovariances.clone()
        
        return 
    
    ###########################################################################
    # UPDATE at superstate level
    
    # Change the cluster assignments to mimick those of another MJPF which needs
    # to be coupled with the current one.
    # INPUTS:
    # - clusterAssignmentsOtherMJPF: cluster assignments of the other MJPF.
    # - timeInClustersOtherMJPF: also the time spent in each cluster is updated
    #      based on that of the other MJPF.
    def UpdateSuperstatesBasedOnOtherMJPF(self, clusterAssignmentsOtherMJPF, timeInClustersOtherMJPF):
        
        for particleIndex in range(self.numberOfParticles):  
            
            self.clusterAssignments[int(particleIndex)] = clusterAssignmentsOtherMJPF[int(particleIndex)]
            self.timeInClusters[int(particleIndex)]     = timeInClustersOtherMJPF[int(particleIndex)]
                    
        return
    
    # Function to look at the updated value and to change the cluster assignments
    # of the particles based on that. This is to avoid having particles that
    # are given to a very far away cluster due to the sequencing.
    def UpdateSuperstatesBasedOnUpdate(self):
        
        for particleIndex in range(self.numberOfParticles):  
            
            currentParticleMean = self.particlesMeansUpdated[:, particleIndex]
            
            minDistance = 100000000
            minIndex    = 0
            
            # searchClosestParticleMean
            for i in range(self.numberOfClusters):
                
                currentClusterMean = self.nodesMean[i,:]
               
                distance = torch.mean(torch.abs(currentParticleMean-currentClusterMean))
                
                if distance.item() < minDistance:
                    minDistance = distance.item()
                    minIndex = i
                    
            self.clusterAssignments[particleIndex] = minIndex
                   
        return
    
    # Function to update the superstates of the particles choosing between
    # the assignments of the current MJPF and that of another one.
    # INPUTS:
    # - clusterAssignmentsOtherMJPF: cluster assignments of the other MJPF.
    # - timeInClustersOtherMJPF: also the time spent in each cluster is updated
    #      based on that of the other MJPF (in case the cluster of the other 
    #      MJPF is chosen).
    def UpdateSuperstatesBasedOnUpdateTwoPossibilities(self, clusterAssignmentsOtherMJPF, timeInClustersOtherMJPF):
        
        for particleIndex in range(self.numberOfParticles):  
            
            currentParticleMean            = self.particlesMeansUpdated[:, particleIndex]
            
            currentAssignment              = self.clusterAssignments[int(particleIndex)]
            currentAssignmentMean          = self.nodesMean[int(currentAssignment),:]
            distance                       = torch.mean(torch.abs(currentParticleMean-currentAssignmentMean))
            
            currentAssignmentOtherMJPF     = clusterAssignmentsOtherMJPF[int(particleIndex)]
            currentAssignmentMeanOtherMJPF = self.nodesMean[int(currentAssignmentOtherMJPF),:]
            distanceOtherMJPF              = torch.mean(torch.abs(currentParticleMean-currentAssignmentMeanOtherMJPF))
            
            if distanceOtherMJPF.item() < distance.item():
                self.clusterAssignments[particleIndex] = currentAssignmentOtherMJPF
                self.timeInClusters[particleIndex]     = timeInClustersOtherMJPF
                    
        return
                
    ###########################################################################
    # PREDICTION at state level
    
    # Function to perform prediction of mean and covariance for all the particles,
    # using the learned prediction models.
    def PredictParticlesMeansAndCovariances(self):
        
        # Looping over the number of particles
        for particleIndex in range(self.numberOfParticles):  
            
            predictionMean, predictionCov = self.PredictParticle(particleIndex)
            
            self.particlesMeans[:, particleIndex]          = torch.squeeze(predictionMean.clone())
            self.particlesCovariances[:, :, particleIndex] = predictionCov.clone()
            
        self.particlesMeansPredicted       = self.particlesMeans.clone()
        self.particlesCovariancesPredicted = self.particlesCovariances.clone()
        
        return
    
    ###########################################################################
    # PREDICTION at superstate level
    
    # Function to predict the next cluster, for a particle identified by its
    # index.
    # This function combines two models:
    # - the general transition matrix;
    # - the temporal transition matrices;
    def PerformSuperstatePredictionParticleAtGivenIndex(self, particleIndex):
        
        predicted_superstate_previous = self.clusterAssignments[particleIndex]
        
        # Select row of transition matrix
        transitionMatRow              = self.transitionMat[int(predicted_superstate_previous),:]
        
        # Select row of vicinity transition matrix
        if self.vicinityTransitionMat is not None:
            print('THERE IS VICINITY MAT')
            vicinityTransitionMatRow      = self.vicinityTransitionMat[int(predicted_superstate_previous),:]
        
        startingRatioForMoveToAll = 1
        
        # If we have temporal transition matrices too
        if self.maxClustersTime.all != None:
            
            # Considering time matrices, if we have been in a cluster for
            # more than one time instant
            maxTimeCurrentCluster = self.maxClustersTime[int(predicted_superstate_previous)]-1
            
            # Time passed in this cluster
            timeInThisCluster     = self.timeInClusters[particleIndex]
            
            if timeInThisCluster > 1 and timeInThisCluster < maxTimeCurrentCluster*startingRatioForMoveToAll: # be careful to use < instead of <=
                # select the temporal transition matrix related to being
                # in the current cluster for timeInClusters[particleIndex] instances
                if timeInThisCluster > maxTimeCurrentCluster:
                    curr_temporalTransitionMatrix = self.temporalTransitionMatrix[int(maxTimeCurrentCluster)]
                else:
                    curr_temporalTransitionMatrix = self.temporalTransitionMatrix[int(timeInThisCluster)]
                temporalTransitionMatRow      = curr_temporalTransitionMatrix[int(predicted_superstate_previous),:]
                
                if self.vicinityTransitionMat is not None:
                    finalTransitionMatRow         = temporalTransitionMatRow + transitionMatRow + vicinityTransitionMatRow*self.transitionMatExplorationRatio
                else:
                    finalTransitionMatRow         = temporalTransitionMatRow + transitionMatRow
                finalTransitionMatRow         = finalTransitionMatRow/torch.sum(finalTransitionMatRow)
                
            elif timeInThisCluster > 1 and timeInThisCluster >= np.floor(maxTimeCurrentCluster)*startingRatioForMoveToAll:
                
                curr_temporalTransitionMatrix = self.temporalTransitionMatrix[int(maxTimeCurrentCluster)]
                temporalTransitionMatRow      = curr_temporalTransitionMatrix[int(predicted_superstate_previous),:]
                if self.vicinityTransitionMat is not None:
                    finalTransitionMatRow         = temporalTransitionMatRow + transitionMatRow + vicinityTransitionMatRow*self.transitionMatExplorationRatio
                else:
                    finalTransitionMatRow         = temporalTransitionMatRow + transitionMatRow
                finalTransitionMatRow         = finalTransitionMatRow/torch.sum(finalTransitionMatRow)

            else:
                if self.vicinityTransitionMat is not None:
                    finalTransitionMatRow = transitionMatRow + vicinityTransitionMatRow*self.transitionMatExplorationRatio
                else:
                    finalTransitionMatRow = transitionMatRow.clone()
                finalTransitionMatRow         = finalTransitionMatRow/torch.sum(finalTransitionMatRow)
        else:
            if self.vicinityTransitionMat is not None:
                finalTransitionMatRow = transitionMatRow + vicinityTransitionMatRow*self.transitionMatExplorationRatio
            else:
                finalTransitionMatRow = transitionMatRow.clone()
            finalTransitionMatRow         = finalTransitionMatRow/torch.sum(finalTransitionMatRow)
                
        
        # Perform superstate prediciton based on probability from transition 
        # matrix / transition matrices
        predicted_superstate_current           = random.choices(np.arange(self.numberOfClusters), weights = finalTransitionMatRow, k = 1)[0]
        
        # Change cluster assignment
        self.clusterAssignments[particleIndex] = predicted_superstate_current
        
        # Did cluster change?
        self.ChangeTimeInClusterParticleAtGivenIndex(particleIndex, predicted_superstate_previous, predicted_superstate_current)
        
        self.passagesBetweenClusters[particleIndex, 0] = int(predicted_superstate_previous)
        self.passagesBetweenClusters[particleIndex, 1] = int(predicted_superstate_current)
        
        return

    # Function to change the value of the time spent in a particle (if it has
    # changed).
    # INPUTS:
    # - particleIndex: index of the particle.
    # - predicted_superstate_previous: superstate of particle in previous time instant
    # - predicted_superstate_current: superstate of particle in curernt time instant
    def ChangeTimeInClusterParticleAtGivenIndex(self, particleIndex, predicted_superstate_previous, predicted_superstate_current):
        
        if predicted_superstate_previous == predicted_superstate_current:
            # If same superstate, add 1
            self.timeInClusters[particleIndex] = self.timeInClusters[particleIndex] + 1                                          
        else:
            # Else rinizialize 
            self.timeInClusters[particleIndex] = 0                                                
        
        return
    
    # Function to perform prediction of the next superstate, for all the particles.
    def PerformSuperstatesPrediction(self):
        
        for particleIndex in range(self.numberOfParticles):   
            self.PerformSuperstatePredictionParticleAtGivenIndex(particleIndex)
            
        self.CalculatePredictedParticlesHistogram()
        
        return
    
    ###########################################################################
    # KLDA calculation
    
    # Function to calculate the histogram of the predicted particles.
    # This function updates the internal variable 'histogramPredictedParticles'.
    def CalculatePredictedParticlesHistogram(self):
        
        histogram = np.zeros(self.numberOfClusters)
        
        for particleIndex in range(self.numberOfParticles):
            active_Node_sample = np.where(self.clusterAssignments == particleIndex)[0]
            if active_Node_sample.size != 0:
                weight                   = self.particlesWeights[active_Node_sample[0]]
                histogram[particleIndex] = np.sum(active_Node_sample)*weight
        
        histogram = histogram/(np.sum(histogram))
        
        self.histogramPredictedParticles = histogram
        
        return
        
    # This function calculates the overall KLDA, combining:
    # - probability of each cluster, from observations;
    # - predicted clusters, based on the transition matrix, calculated using
    #   each particle and the histogram of the particles.
    # INPUTS:
    # - probabilitiesOfClusters: the probability of being in each cluster.
    # OUTPUTS:
    # - KLDA: the calculated Kullback-Leibler Divergence Anomaly
    def CalculateOverallKLDA(self, probabilitiesOfClusters):
        
        KLDA = d_utils.KLD_Abnormality(self.numberOfClusters, self.numberOfParticles, 
                                       self.histogramPredictedParticles, self.transitionMat, 
                                       probabilitiesOfClusters, self.KLDAbnMax)
        
        if KLDA > self.KLDAbnMax:
            self.KLDAbnMax = KLDA
        
        return KLDA
    
    # This function calculates the KLDA for a single particle, for all particles,
    # combining:
    # - probability of each cluster, from observations;
    # - predicted clusters, based on the transition matrix, for the particle
    #   considered.
    # INPUTS:
    # - probabilitiesOfClusters: the probability of being in each cluster.
    # OUTPUTS:
    # - KLDA: the calculated Kullback-Leibler Divergence Anomaly
    def CalculateKLDAForEachParticle(self, probabilitiesOfClusters):
        
        KLDAs = torch.zeros(self.numberOfParticles)
        
        for particleIndex in range(self.numberOfParticles): 
            
            currentParticleKLDA  = self.CalculateKLDAForSingleParticle(probabilitiesOfClusters, particleIndex)
            
            # Insert KLDA value of current particle in the array of KLDAs
            KLDAs[particleIndex] = currentParticleKLDA
        
        return KLDAs
    
    # This function calculates the KLDA for a single particle.
    # combining:
    # - probability of each cluster, from observations;
    # - predicted clusters, based on the transition matrix, for the particle
    #   considered.
    # INPUTS:
    # - probabilitiesOfClusters: the probability of being in each cluster.
    # - particleIndex: the index of the considered particle.
    # OUTPUTS:
    # - KLDA: the calculated Kullback-Leibler Divergence Anomaly
    def CalculateKLDAForSingleParticle(self, probabilitiesOfClusters, particleIndex):
        
        particleAssignment = self.clusterAssignments[particleIndex]
        
        PP   = torch.squeeze(self.transitionMat[int(particleAssignment),:]) +1e-20 # add 1e-100 since KLD doesnt allow zero values
        QQ   = torch.squeeze(probabilitiesOfClusters)
        
        KLDA = d_utils.single_KLD_Abnormality(PP = PP, QQ = QQ, KLDAbnMax = self.KLDAbnMax, 
                                              histogramProb = 1, N = self.numberOfClusters)
        
        return KLDA
    
    ###########################################################################
    # Reweighting and resampling of particles
    
    # Function to reweight the particles given a given set of probabilities
    # of the particles themselves.
    # INPUTS:
    # - probabilitiesOfParticles: array of probabilities of each particle.
    def ReweightParticles(self, probabilitiesOfParticles):
        
        self.particlesWeights = probabilitiesOfParticles.clone()
        
        return
    
    # Function to resample the particles given the indices of the particles
    # to take.
    # INPUTS:
    # - newIndicesForSwapping: array of indices of all the particles to pick.
    def ResampleParticlesGivenNewIndices(self, newIndicesForSwapping):
        
        # Perform swapping       
        # First create temporary vectors by copying from the ones already present
        clusterAssignmentsTemp              = copy.deepcopy(self.clusterAssignments)
        particlesMeansTemp                  = self.particlesMeans.clone()
        particlesCovariancesTemp            = self.particlesCovariances.clone()
        timeInClustersTemp                  = self.timeInClusters.clone()
        particlesMeansUpdatedTemp           = self.particlesMeansUpdated.clone()
        particlesCovariancesUpdatedTemp     = self.particlesCovariancesUpdated.clone()
        particlesMeansPredictedTemp         = self.particlesMeansPredicted.clone()
        particlesCovariancesPredictedTemp   = self.particlesCovariancesPredicted.clone()
        
        # Change using the temporary vector (so to not generate problems)
        for particleIndex in range(self.numberOfParticles):  
            
            currentSwapIndex = newIndicesForSwapping[particleIndex]
            
            clusterAssignmentsTemp[particleIndex]                 = self.clusterAssignments[currentSwapIndex]
            particlesMeansTemp[:, particleIndex]                  = self.particlesMeans[:, currentSwapIndex]
            particlesCovariancesTemp[:,:,particleIndex]           = self.particlesCovariances[:, :, currentSwapIndex]  
            timeInClustersTemp[particleIndex]                     = self.timeInClusters[currentSwapIndex]
            particlesMeansUpdatedTemp[:, particleIndex]           = self.particlesMeansUpdated[:, currentSwapIndex]
            particlesCovariancesUpdatedTemp[:,:,particleIndex]    = self.particlesCovariancesUpdated[:, :, currentSwapIndex] 
            particlesMeansPredictedTemp[:, particleIndex]         = self.particlesMeansPredicted[:, currentSwapIndex]
            particlesCovariancesPredictedTemp[:,:,particleIndex]  = self.particlesCovariancesPredicted[:, :, currentSwapIndex] 
            
        # Then save the final results from the temporary vector
        self.clusterAssignments              = copy.deepcopy(clusterAssignmentsTemp)
        self.particlesMeans                  = particlesMeansTemp.clone()
        self.particlesCovariances            = particlesCovariancesTemp.clone()
        self.timeInClusters                  = timeInClustersTemp.clone()
        self.particlesMeansUpdated           = particlesMeansUpdatedTemp.clone()
        self.particlesCovariancesUpdated     = particlesCovariancesUpdatedTemp.clone()
        self.particlesMeansPredicted         = particlesMeansPredictedTemp.clone()
        self.particlesCovariancesPredicted   = particlesCovariancesPredictedTemp.clone()
            
        # Weights of the particles
        # Give same weigth to all the particles
        self.particlesWeights      = torch.ones(self.numberOfParticles)/self.numberOfParticles 

        return
    
    # Function to resample the particles. It uses the calculated weights in
    # 'particlesWeights' to pick the indices of the particles to keep.
    # OUTPUTS:
    # - newIndicesForSwapping: the indices used for resampling.
    def ResampleParticles(self):
        
        # Resampling. Save here the new indices.
        newIndicesForSwapping = random.choices(np.arange(self.numberOfParticles), 
                                               weights = self.particlesWeights, 
                                               k       = self.numberOfParticles)
        
        self.ResampleParticlesGivenNewIndices(newIndicesForSwapping)
            
        return newIndicesForSwapping
    
    # Function to resample the particles using the likelihoods ( higher likelihood
    # means higher probability to resample).
    # INPUTS:
    # - likelihoods: an array of likelihoods, a value for each particle
    def ReweightParticlesBasedOnLikelihood(self, likelihoods):
        
        self.particlesWeights = self.particlesWeights*likelihoods
        sum_probabilities     = torch.sum(self.particlesWeights)
        self.particlesWeights = self.particlesWeights/sum_probabilities
        
        return
    
    # Function to resample the particles using an anomaly value (higher anomaly
    # means less probability to resample).
    # INPUTS:
    # - anomalies: an array of anomalies, a value for each particle
    def ReweightParticlesBasedOnAnomalyAndLikelihoods(self, anomalies, likelihoods):
        
        probabilities = torch.zeros(self.numberOfParticles)
        
        for i in range(self.numberOfParticles):
            probabilities[i]  = 1/anomalies[i]
            
        sum_probabilities     = torch.sum(probabilities)
        probabilities         = probabilities/sum_probabilities
        
        self.particlesWeights = self.particlesWeights*probabilities*likelihoods
        sum_probabilities     = torch.sum(self.particlesWeights)
        self.particlesWeights = self.particlesWeights/sum_probabilities
        
        return
    
    # Function to calculate effective sample size.
    # This value goes from 1 to the number of particles of the MJPF.
    # A lower value corresponds to a higher degeneracy of the MJPF.
    # OUTPUTS:
    # - effectiveSampleSize
    def ReweightParticlesBasedOnAnomaly(self, anomalies):
        
        probabilities = torch.zeros(self.numberOfParticles)
        
        for i in range(self.numberOfParticles):
            probabilities[i]  = 1/anomalies[i]
            
        sum_probabilities     = torch.sum(probabilities)
        probabilities         = probabilities/sum_probabilities
        
        self.particlesWeights = self.particlesWeights*probabilities
        sum_probabilities     = torch.sum(self.particlesWeights)
        self.particlesWeights = self.particlesWeights/sum_probabilities
        
        return
    
    # Function to calculate effective sample size.
    # This value goes from 1 to the number of particles of the MJPF.
    # A lower value corresponds to a higher degeneracy of the MJPF.
    # OUTPUTS:
    # - effectiveSampleSize
    def CalculateEffectiveSampleSize(self):
        
        sumNsSquared = 0
        
        for particleIndex in range(self.numberOfParticles):
            
            currentParticleWeight = self.particlesWeights[particleIndex]
            sumNsSquared          += np.power(currentParticleWeight,2)
            
        #print(sumNsSquared)
        effectiveSampleSize = 1/sumNsSquared
        
        return effectiveSampleSize
    
    # Check if resampling is necessary, based on whether the effective sample
    # size is above or below the defined threshold.
    # When it is below the threshold, the degeneracy is too high and resampling
    # will be necessary.
    # OUTPUTS:
    # - True if resampling is necessary (effective sample size below threshold)
    #   and False if otherwise.
    def CheckIfResamplingIsNecessary(self):
        
        effectiveSampleSize = self.CalculateEffectiveSampleSize()
        
        print('Effective sample size: ' + str(effectiveSampleSize))
        
        if effectiveSampleSize < self.resamplingThreshold:
            return True
        else:
            return False
    
    # Function to resample the particles based on an anomaly associated to each 
    # particle itself.
    # INPUTS:
    # - anomalies: a list/1D array composed by one anomaly value for each particle.
    # OUTPUTS:
    # - newIndicesForSwapping: indices of swapped particles
    def ReweightAndResampleParticlesBasedOnAnomaly(self, anomalies):
        
        self.ReweightParticlesBasedOnAnomaly(anomalies)
        
        newIndicesForSwapping = self.ResampleParticles()
        
        return newIndicesForSwapping
    
    # This function allows a different type of resampling w.r.t.
    # the traditional "resampleParticles".
    # "Resample particles" uses a probability over the particles and resamples
    # based on that. 
    # If, for example, the probability of each cluster is used for resampling, 
    # this can lead very quickly to having only particles belonging to a single
    # class! This happens because of this:
    # > Clusters have probabilities 0.4, 0.35, 0.35
    # > Particles get resampled based on the probabilities, so slightly more
    #   particles are kept of cluster 1.
    # > Then the same probabilities are found in the next time step.
    # > Again, some particles of the two least present classes disappear.
    # > And so on... at the end, after few time instants, only the particles
    #   belonging to the most probable cluster survives.
    #
    # With this function, instead, the proportion of the particles depends
    # on their cluster belonging, so, if there are particles belonging to 
    # cluster 1,2, and 3, they are resampled so to keep the proportion given
    # by the cluster probabilities 0.4, 0.35, 0.35.
    #
    # The function performs the following steps:
    # 1> First we check which clusters are present in the current 
    #   clusterAssignments vector, so to reweight the probabilities based on that
    #   (cutting of clusters that are not present)
    # 2> Than, a vector is created holding the position of the particles for each
    #   cluster.
    # 3> Than, based on the cluster probabilities, we pick a cluster and we pick
    #   a particle related to that cluster.
    def ResampleParticlesBasedOnClusterProbabilities(self, clusterProbabilities):
        
        clusterProbabilities = torch.squeeze(clusterProbabilities)
        
        self.clusterProbabilities = clusterProbabilities
        
        # 1> Checking which clusters are present in the current cluster assignment
        # and keeping those probabilities.
        # clusterProbabilitiesNew will contain the probabilities of those clusters
        # that are actually present in the clusterAssignment vector, while all 
        # else will be zero.
        clusterProbabilitiesNew = np.zeros(len(clusterProbabilities))
        for i in range(self.numberOfParticles):
            current_cluster                               = self.clusterAssignments[i]
            clusterProbabilitiesNew[int(current_cluster)] = clusterProbabilities[int(current_cluster)]
        # Normalizing
        clusterProbabilitiesNew = clusterProbabilitiesNew/np.sum(clusterProbabilitiesNew)
        
        self.clusterProbabilitiesNew = clusterProbabilitiesNew
        
        # 2> Now create a vector holding the location of the particles for each cluster.
        indicesParticlesPerCluster = []
        for i in range(self.numberOfClusters):
            currentClusterIndices = []
            indicesParticlesPerCluster.append(currentClusterIndices)
        for i in range(self.numberOfParticles):
            # Current cluster of the particle
            current_cluster                          = self.clusterAssignments[i]
            # Put the index of the particle in the corresponding cluster list
            indicesParticlesPerCluster[int(current_cluster)].append(i)
        
        self.indicesParticlesPerCluster = indicesParticlesPerCluster
            
        # 3> Now pick the particles to keep
        # Which clusters to pick?
        clustersToPick = random.choices(np.arange(self.numberOfClusters), 
                                        weights = clusterProbabilitiesNew, 
                                        k       = self.numberOfParticles)
        self.clustersToPick = clustersToPick
        # Which particles to keep based on cluster belonging?
        newIndicesForSwapping = np.zeros(self.numberOfParticles)
        for i in range(self.numberOfParticles):
            # Cluster of the particle
            current_cluster          = clustersToPick[i]
            # Pick particle
            current_particle         = random.choice(indicesParticlesPerCluster[int(current_cluster)])
            newIndicesForSwapping[i] = int(current_particle)
        
        newIndicesForSwapping = newIndicesForSwapping.astype(int)
        
        # Now resample given the indices
        self.ResampleParticlesGivenNewIndices(newIndicesForSwapping)

        return newIndicesForSwapping

    ###########################################################################
    # For case with other MJPF combined
    
    # Function to find the nearest neighboor of a value from a set of array values.
    # INPUTS:
    # - array: (state_dim*numParticles)
    # - values: (state_dim)
    @staticmethod
    def find_nearest(array, value):
        value_repeated = torch.unsqueeze(value, 1)
        
        if array.ndim == 1:
            array = torch.unsqueeze(array, 1)
            
        value_repeated = value_repeated.repeat(1, array.shape[1])
        
        difference     = torch.mean(torch.abs(array-value_repeated), 0)       
        idx            = torch.argmin(difference)

        return int(idx), array[:,idx].clone(), difference[idx].clone()
    
    # Function to find which is the closest particle of a second MJPF from the
    # particles of the current MJPF.
    # INPUTS:
    # - clusterAssignmentOfOtherMJPF
    # - particle index
    # OUTPUTS:
    # - the closest cluster
    def FindClosestClusterOfSecondMJPFSingleParticle(self, clusterAssignmentOfOtherMJPF, particleIndex):
        
        singleClusterAssignment    = self.clusterAssignments[particleIndex]
        numberOfParticlesOtherMJPF = len(clusterAssignmentOfOtherMJPF)
        
        # First look if the same cluster is present (this is often the case, so
        # it can save time).
        if singleClusterAssignment in clusterAssignmentOfOtherMJPF:
            return singleClusterAssignment
        # If it is not the case, let us look for the closest cluster
        else:
            
            # Set initial closest cluster to same cluster and a very big cluster distance.
            closestCluster          = singleClusterAssignment
            smallestClusterDistance = 100000000
            
            # Looping over the particles of the other MJPF
            for i in range(numberOfParticlesOtherMJPF):
                
                # Current cluster of other MJPF
                currentClusterAssignmentOtherMJPF = clusterAssignmentOfOtherMJPF[i]
                currentClusterDistance            = self.nodesVicinityMatrix[int(singleClusterAssignment),
                                                                             int(currentClusterAssignmentOtherMJPF)] 
                # If it was closer, then assign it as closest cluster
                if currentClusterDistance.item() < smallestClusterDistance:
                    closestCluster          = currentClusterAssignmentOtherMJPF
                    smallestClusterDistance = currentClusterDistance.item()
                    
            return closestCluster
    
    ###########################################################################
    # For anomalies
    
    # This function allows to find an anomaly between the prediction at state 
    # level and an externally given value for the state.
    # This value could be an observation from the sensors.
    # For example, in the case of the MJPF on odometry, the external value 
    # could be the observation from the GPS or from the lidar odometry.
    # Anomaly is simply the MSE between the two values, without including the
    # covariance information.
    # INPUTS:
    # - externalValue
    # OUTPUTS:
    # - anomalies
    def FindStatePredictionVsSingleExternalValueAnomalyMSE(self, externalValue):
        
        # Initialize anomaly vector
        anomalies = torch.zeros(self.numberOfParticles).to(device)
        
        # Looping over the number of particles
        for particleIndex in range(self.numberOfParticles):  
            
            currentParticlePrediction  = self.particlesMeansPredicted[:,particleIndex]
            
            # Calculate anomaly
            currentAnomaly             = F.mse_loss(currentParticlePrediction[0:2], 
                                                    externalValue[0:2],size_average=False)   
            
            anomalies[particleIndex]   = currentAnomaly

        return anomalies
    
    # This function allows to find an anomaly between the prediction at state 
    # level for each particle and an external value, one for each particle.
    # In the case of the odometry from video code, the external value could be
    # the odometry prediction performed by the video module.
    # Anomaly is simply the MSE between the two values, without including the
    # covariance information.
    # INPUTS:
    # - externalValue
    # OUTPUTS:
    # - anomalies
    def FindStatePredictionVsExternalValuesAnomalyMSE(self, externalValues):
        
        # Initialize anomaly vector
        anomalies = torch.zeros(self.numberOfParticles).to(device)
        
        # Looping over the number of particles
        for particleIndex in range(self.numberOfParticles):  
            
            currentParticlePrediction  = self.particlesMeansPredicted[:,particleIndex]
            currentExternalValues      = torch.squeeze(externalValues[:,particleIndex])
            
            # Calculate anomaly
            currentAnomaly             = F.mse_loss(currentParticlePrediction, 
                                                    currentExternalValues,size_average=False)   
            
            anomalies[particleIndex]   = currentAnomaly

        return anomalies
    
    # Find the likelihood of each particle comparing its mean and covariance
    # value with the means and covariances of the clusters of a cluster graph.
    # INPUTS:
    # - clusterGraph: a given clustering object.
    # - temperature: temperature of the assignment.
    # OUTPUTS:
    # - particlesLikelihoodGivenCluster    
    def ObtainParticleLikelihoodGivenClusterGraph(self, clusterGraph, temperature = 1):
        
        # Obtain cluster likelihoods
        particlesClusterProbabilities = torch.zeros(self.numberOfParticles, self.numberOfClusters)
        
        # Looping over the number of particles
        for i in range(self.numberOfParticles):
            
            currentParticleMean     = self.particlesMeansUpdated[:,i]
            currentParticleCov      = self.particlesCovariancesUpdated[:,:,i]
            currentParticleDistance = clusterGraph.FindBhattaDistanceFromClustersSinglePointGivenCovariance(
                                      currentParticleMean,currentParticleCov)
            
            particlesClusterProbabilities[i,:] = d_utils.CalculateProbabilitiesFromDistances(
                                                 currentParticleDistance, temperature)
            
        # obtain particle likelihoods given clusters
        particlesLikelihoodGivenCluster = torch.zeros(self.numberOfParticles)
        
        # Looping over the number of particles
        for i in range(self.numberOfParticles):
            
            clusterAssignment                  = self.clusterAssignments[i]
            particlesLikelihoodGivenCluster[i] = particlesClusterProbabilities[i,int(clusterAssignment)]
            
        particlesLikelihoodGivenCluster = particlesLikelihoodGivenCluster/torch.sum(particlesLikelihoodGivenCluster)
            
        return particlesLikelihoodGivenCluster
    
    # Find the likelihood of the particles given the probability of each cluster.
    # INPUTS:
    # - clusterProbabilities: probability of each cluster.
    # OUTPUTS:
    # - particlesLikelihoodGivenCluster
    def ObtainParticleLikelihoodGivenProbabilitiesOfClusters(self, clusterProbabilities):
        
        # obtain particle likelihoods given clusters
        particlesLikelihoodGivenCluster = torch.zeros(self.numberOfParticles)
        
        # Looping over the number of particles
        for i in range(self.numberOfParticles):
            
            clusterAssignment                  = self.clusterAssignments[i]
            if clusterProbabilities.ndim == 2:
                likelihoodOfCluster                = clusterProbabilities[0,int(clusterAssignment)]
            elif clusterProbabilities.ndim == 1:
                likelihoodOfCluster                = clusterProbabilities[int(clusterAssignment)]
            particlesLikelihoodGivenCluster[i] = likelihoodOfCluster
            
        particlesLikelihoodGivenCluster = particlesLikelihoodGivenCluster/torch.sum(particlesLikelihoodGivenCluster)
        
        return particlesLikelihoodGivenCluster