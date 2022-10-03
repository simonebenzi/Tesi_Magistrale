# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 13:38:37 2021

@author: giulia
"""


import numpy as np
import numpy.matlib
from sympy import difference_delta
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import scipy.stats
from torchvision.utils import save_image
import unittest
from navpy import quat2angle

#from ConfigurationFiles import Config_GPU as ConfigGPU 

###############################################################################
# ----------------------------- GPU setting  ----------------------------------
###############################################################################
# GPU or CPU?
#configGPU = ConfigGPU.ConfigureGPUSettings()
#device    = ConfigGPU.DefineDeviceVariable(configGPU)

###############################################################################
# -------------------------  KVAE odometry from video--------------------------
###############################################################################

###############################################################################
# Exception classes

class NonExistentTypeOfWeighting(Exception):

    def __init__(self, type_of_weighting):
        self.type_of_weighting = type_of_weighting 
    def __str__(self):
        message = 'The type of weighting {} does not momentarely exist in the code. '.format(self.type_of_weighting) + \
                  'Either change the value in the configuration file ' + \
                  '<Config_MJPFs_running_functions>, or insert it in the code.'
        return message
    
###############################################################################
# KVAE class that, given a built KVAE uses it to perform filtering/smoothing
# to track the evolution of odometry from video

# To note: this contains a KVAE object, but does not inherit from it.

class KVAE_odometry_from_video(nn.Module):
    
    ###########################################################################
    # INITIALIZATION
    
    def FindAnomalyThresholds(self):
        
        self.anomalyThresholds = self.anomaliesMeans + self.stdTimes*self.anomaliesStandardDeviations     
        return
    
    def FindTimeEnough(self, time_enough_ratio):
        
        self.time_enough = time_enough_ratio*self.time_window
        return
    
    def FindTimeWait(self, time_wait_ratio):
        
        self.time_wait = time_wait_ratio*self.time_window
        return
    
    def FindDerivatedTimeWindows(self, time_enough_ratio, time_wait_ratio):
        
        self.FindTimeEnough(time_enough_ratio)
        self.FindTimeWait(time_wait_ratio)
        return
    
    # Function to initialize the model for obtaining odometry from video
    # INPUTS:
    # - kvae: loaded kvae
    # - clusterGraphVideo: cluster graph object of video (state 'a')
    # - clusterGraphVideoZ: cluster graph object of video (state 'z')
    # - clusterGraphParams: cluster graph object of odometry
    # - mjpf: mjpf over odometry
    # - mjpf_video: mjpf over video (it follows the same assignments as the one
    #   of odometry).
    # - anomaliesMeans and anomaliesStandardDeviations: two numpy arrays, one
    #   with the mean values of the anomalies over training, and the other one
    #   with the standard deviations.
    # - time_window, time_enough_ratio, time_wait_ratio: parameters for choosing
    #   when to restart a subset of the particles.
    #   > time_window defines the length of a time window over which to calculate
    #   the anomalies;
    #   > time_enough_ratio is the ratio of 'time_window' sufficient to fire
    #   a restarting for standard anomalies. However, some anomalies, when over
    #   the threshold would need for a delayed restart (e.g., reconstruction
    #   anomaly).
    #   time_enough = time_enough_ratio*time_window
    #   > time_wait_ratio is the ratio of 'time_window' to wait before restarting
    #   the subset of particles, after an anomaly of length higher than 
    #   'time_enough' is detected on one of the 'waiting' anomalies. 
    #   time_wait = time_wait_ratio*time_window
    # - usingAnomalyThresholds: are we using the anomaly thresholds and performing
    #   restarts. If set to False, no restarting of subsets of particles is 
    #   performed.
    def __init__(self, kvae, clusterGraphVideo, clusterGraphVideoZ, clusterGraphParams, 
                 mjpf, mjpf_video, skew_video, anomaliesMeans = None, anomaliesStandardDeviations = None, 
                 time_window = 20, time_enough_ratio = 0.6, time_wait_ratio = 0.25, stdTimes = 2.5, 
                 usingAnomalyThresholds = False, firstResampleThresh = 0.001, resampleThresh = 0.6):
        super(KVAE_odometry_from_video, self).__init__()
        
        self.kvae                 = kvae
        self.clusterGraphVideo    = clusterGraphVideo
        self.clusterGraphVideoZ   = clusterGraphVideoZ
        self.clusterGraphParams   = clusterGraphParams
        # Previous time instant variables
        self.alphaPrev            = None
        self.Sigma_pred           = None
        self.mu_pred              = None
        self.alpha_prev           = None        
        # PARTICLE FILTER 
        self.mjpf                 = mjpf
        self.mjpf_video           = mjpf_video
        # Skew video
        self.skew_video           = skew_video
        
        # Number of anomalies
        # 0: KLDA anomaly
        # 1: video prediction anomaly
        # 2: video reconstruction anomaly
        # 3: cluster likelihoods anomalies
        # 4: anomaly on the difference between the two predictions
        #    (from video and from odometry dynamics)
        self.numberOfAnomalies = 5
        # Anomalies for which it is necessary to wait before restarting
        # e.g.: it does not make sense to restart when reconstruction 
        # anomaly is high: we will very likely restart from a wrong zone!
        self.indicesWaitingAnomalies = []
        self.indicesWaitingAnomalies.append(2)
        # Initialize anomaly windows
        self.InitializeAnomalyWindows()

        # First odometry value
        self.firstOdometryValue = None

        # First orientation value
        self.firstOrientationValue = None
        
        # Restarting information
        self.usingAnomalyThresholds = usingAnomalyThresholds
        self.anomaliesMeans = anomaliesMeans
        self.anomaliesStandardDeviations = anomaliesStandardDeviations
        self.stdTimes = stdTimes
        self.time_window = time_window
        self.FindDerivatedTimeWindows(time_enough_ratio, time_wait_ratio)
        if self.usingAnomalyThresholds == True:
            self.FindAnomalyThresholds()            
        self.whyRestarted = []
        self.whereRestarted = []        
        self.needReinit = False
        self.needReinitAfterItIsGoodAgain = False

        self.IMUInstant = 0

        # IMU estimation
        if self.mjpf != None:
            self.estimatedOdometryFromIMU = np.zeros((4, self.mjpf.numberOfParticles))
        
        # Resampling thresholds
        if self.mjpf != None:
            self.firstResampleThresh = firstResampleThresh
            self.resampleThresh = resampleThresh*self.mjpf.numberOfParticles

        return
    
    ###########################################################################
    # RE-INITIALIZATION
    
    # Function to re-assign parameters of the combined MJPF object from a 
    # configuration holder object.
    # INPUTS:
    # - configHolder: configuration holder object
    def ReassignParametersFromConfigHolder(self, configHolder):
        
        self.mjpf_video.numberOfParticles = int(configHolder['N_Particles'])
        self.mjpf.numberOfParticles = int(configHolder['N_Particles'])
        self.firstResampleThresh = configHolder['firstResampleThresh']
        self.resampleThresh = configHolder['resampleThresh']*self.mjpf.numberOfParticles
        self.mjpf_video.resamplingThreshold = self.firstResampleThresh
        self.mjpf.resamplingThreshold = self.firstResampleThresh
        
        self.mjpf_video.RestartVariables()
        self.mjpf.RestartVariables()
        self.skew_video = configHolder['skew_video']
        # Time windows
        self.time_window = configHolder['time_window']
        time_enough_ratio = configHolder['time_enough_ratio']
        time_wait_ratio = configHolder['time_wait_ratio']
        self.FindDerivatedTimeWindows(time_enough_ratio, time_wait_ratio)
        
        self.FindAnomalyThresholds()
        
        return
    
    ###########################################################################
    # Distance from clusters functions
    
    # Find the value of alpha for video data (cluster of 'a' state)
    # INPUTS:
    # - distances: distances of video points from video clusters
    # - alphaPrev: previous value of alpha
    def alphaDistProbVideo(self, distances, alphaPrev = None, timeInClusters = None, previousMaxAlphas = None):
        skew = self.skew_video
        return self.kvae.alphaDistProbGivenSkew(distances, skew, alphaPrev, timeInClusters, previousMaxAlphas)
    
    # This function finds the distances of a sequence of data from the 
    # mean of the clusters, using the VIDEO clusters and so the ENCODED STATE 'a'
    # obtained after encoding with the VAE.
    # INPUTS:
    # - videoStateData: the encoded state 'a' of the data
    # OUTPUTS:
    # - the distances of each encoded state 'a' from the K video cluster centers.
    def FindDistancesFromVideoClusters(self, videoStateData):
    
        # Extract the Bhattacharya distances of the datapoints from the 
        # clusters of the graph
        distances         = self.clusterGraphVideo.FindDistancesFromVideoClustersNoCov(
                                               videoStateData)
        return distances

    # This function finds the distances of a sequence of data from the 
    # mean of the clusters, using the VIDEO clusters and so the ENCODED STATE 'z'
    # obtained after encoding with the VAE.
    # INPUTS:
    # - videoStateData: the encoded state 'z' of the data
    # OUTPUTS:
    # - the distances of each encoded state 'z' from the K video cluster centers.
    def FindDistancesFromVideoClustersZ(self, videoStateData):

        # Extract the Bhattacharya distances of the datapoints from the 
        # clusters of the graph
        distances         = self.clusterGraphVideoZ.FindDistancesFromVideoClustersNoCov(
                                               videoStateData)        
        return distances
    
    def FindDistancesFromParamsClusters(self, paramsStateData):
        
        distances         = self.clusterGraphParams.FindDistancesFromParamsClustersNoCov(
                                               paramsStateData)        
        return distances
    
    ###########################################################################
    # FUNCTIONS TO HANDLE ANOMALY WINDOWS
    
    # Function operated after restart of a subset of particles.
    def HandleAnomalyWindowingAfterRestart(self):
        
        self.timeAfterReinit = 0
        self.InitializeAnomalyWindows()
        self.reinitialized = False
        self.needReinit = False
        self.needReinitAfterItIsGoodAgain = False
        return
    
    # Initialize the anomaly windows: a list of empty lists.
    def InitializeAnomalyWindows(self):
        
        self.anomalyWindows = []
        for i in range(self.numberOfAnomalies):
            anomalyWindow = []
            self.anomalyWindows.append(anomalyWindow)
        return
    
    # Function used in 'UpdateAnomalyWindows' over a single anomaly.
    def EliminateOldestValueFromSingleAnomalyWindow(self, indexOfAnomaly):
        
        self.anomalyWindows[indexOfAnomaly].pop(0)        
        return
    
    # Eliminate the oldest value from the anomaly windows.
    def EliminateOldestValueFromAnomalyWindows(self):
        
        for indexOfAnomaly in range(self.numberOfAnomalies):
            self.EliminateOldestValueFromSingleAnomalyWindow(indexOfAnomaly)          
        return
    
    # Function used in 'UpdateAnomalyWindows' over a single anomaly.
    def UpdateSingleAnomalyWindow(self, indexOfAnomaly, anomalyValue):
        
        if anomalyValue > self.anomalyThresholds[indexOfAnomaly]:
            self.anomalyWindows[indexOfAnomaly].append(1)
        else:
            self.anomalyWindows[indexOfAnomaly].append(0)            
        return
    
    # Update the anomaly windows, checking whether, for each anomaly, we are
    # above the corresponding threshold.
    # INPUTS:
    # - anomalyVectorCurrentTimeInstant: a vector of as many elements as the
    #   considered number of anomalies for restarting. Each element is either
    #   0 or 1. It is 0 when the anomaly is below its threshold; it is 1 when 
    #   it is above.
    def UpdateAnomalyWindows(self, anomalyVectorCurrentTimeInstant):
        
        for indexOfAnomaly in range(self.numberOfAnomalies):
            anomalyValue = anomalyVectorCurrentTimeInstant[indexOfAnomaly]
            self.UpdateSingleAnomalyWindow(indexOfAnomaly, anomalyValue)        
        return
    
    # Function used in 'CalculateSumsOverAnomalyWindows' over a single anomaly.
    def CalculateSumOverSingleAnomalyWindow(self, indexOfAnomaly):
        
        sumOverSingleAnomalyWindow = sum(self.anomalyWindows[indexOfAnomaly])
        return sumOverSingleAnomalyWindow
    
    # OUTPUTS:
    # - sumsOverAnomalyWindows: an array of as many elements as the number
    #   of anomalies for restarting. Each element contains the count of elements
    #   that went over the anomaly threshold.
    def CalculateSumsOverAnomalyWindows(self):
        
        sumsOverAnomalyWindows = []
        for indexOfAnomaly in range(self.numberOfAnomalies):
            sumAtIndex = self.CalculateSumOverSingleAnomalyWindow(indexOfAnomaly)
            sumsOverAnomalyWindows.append(sumAtIndex)
        return sumsOverAnomalyWindows
    
    def PrintSumsOverAnomalyWindows(self, sumsOverAnomalyWindows):
        
        print('currentSumOverKLDA')
        print(sumsOverAnomalyWindows[0])
        print('currentSumOverVideo')
        print(sumsOverAnomalyWindows[1])
        print('currentSumOverRec')
        print(sumsOverAnomalyWindows[2])
        print('currentSumOverProbs')
        print(sumsOverAnomalyWindows[3])
        print('currentSumOverDiffPreds')
        print(sumsOverAnomalyWindows[4])        
        print('When we restart:')
        print(self.whenRestarted)
        return
    
    # Function to check whether a subset of the particles should be restarted.
    # INPUTS:
    # - sumsOverAnomalyWindows: an array of as many elements as the number
    #   of anomalies for restarting. Each element contains the count of elements
    #   that went over the anomaly threshold.
    def CheckIfParticlesRestartingIsNecessary(self,sumsOverAnomalyWindows):
        
        # First check all the anomalies as if they were of the same kind, i.e.,
        # anomalies that, when, above a threshold, require the restarting 
        # immediately.
        for indexOfAnomaly in range(self.numberOfAnomalies):
            sumAtIndex = sumsOverAnomalyWindows[indexOfAnomaly]
            if sumAtIndex > self.time_enough:
                self.needReinit = True
                self.whyRestarted.append(sumAtIndex)
                print('Restarting is necessary due to anomaly at index {}.'.format(indexOfAnomaly))
                print('We have had more than {} (i.e., {}) anomalies in the last {} time instants.'.format(
                      self.time_enough, sumAtIndex, self.time_window))
        # Then look at the anomalies that require posticipated restarting
        for indexOfAnomaly in self.indicesWaitingAnomalies:
            sumAtIndex = sumsOverAnomalyWindows[indexOfAnomaly]
            if sumAtIndex > self.time_enough:
                self.needReinit = False
                self.whyRestarted.pop()
                self.needReinitAfterItIsGoodAgain = True
                print('... but first wait.')
            if self.needReinitAfterItIsGoodAgain == True and sumAtIndex < self.time_enough - self.time_wait:
                self.needReinitAfterItIsGoodAgain = False
                self.needReinit = True
                print('Waiting ended.')
        return
    
    ###########################################################################
    # FILTERING FUNCTIONS

    # Code 100: performing filtering supposing to know the odometry already.
    # This is to see what the best result could be.
    # INPUTS:
    # - currentImagesBatch: image values of current batch
    # - currentParamsBatch: odometry values of current batch
    # - curr_distance: distances from clusters
    def PerformKVAEFilteringUsingParamsClusteringGivenDistance(self, currentImagesBatch, currentParamsBatch, curr_distance = None):
        
        ################################# VAE #################################
        # ENCODE AND DECODE
        # Performed calling the encoding and decoding method on the VAE object
        reconstructedImagesBatch, a_seq, a_mu, a_var = self.kvae.CallVAEEncoderAndDecoderOverBatchSequence(currentImagesBatch)        
        if curr_distance == None:
            print('Calculate distance')
            _dist = self.FindDistancesFromParamsClusters(torch.unsqueeze(torch.unsqueeze(torch.squeeze(currentParamsBatch), 0), 0))
            _dist = _dist[:,0,:].double()
        else:
            _dist = curr_distance[:,0,:].double()           
        ########################### SMOOTHING #################################
        self.kvae.kf.initializeCallToKalmanFilter(dist=_dist.double(), y=a_seq, alphaDistProb = self.kvae.alphaDistProb)        
        if self.timeInstant == 0:    
            print('FIRST TIME INSTANT')
            self.alpha             = (self.kvae.alphaDistProb(_dist))
            self.kvae.kf.updateAlphaBeginning(self.alpha) 
            self.mu_pred           = self.kvae.kf.mu
            self.Sigma_pred        = self.kvae.kf.Sigma      
        #######################################################################
        # UPDATE PHASE
        mu_t, Sigma_t, C          = self.kvae.kf.perform_update(self.alpha, self.Sigma_pred, self.mu_pred, a_mu[:,0,:])
        input_for_alpha_update    = _dist, self.alpha
        self.alpha                = self.kvae.kf.update_alpha_CG_KVAE(input_for_alpha_update)
        #######################################################################
        # PREDICTION PHASE
        mu_pred, Sigma_pred, A, B = self.kvae.kf.perform_prediction(self.alpha, mu_t, Sigma_t)        
        self.mu_pred    = mu_pred
        self.Sigma_pred = Sigma_pred
        # Time instant increase
        self.timeInstant += 1
     
        return reconstructedImagesBatch, a_seq, a_mu, a_var, self.alpha, mu_t, Sigma_t, mu_pred, Sigma_pred, A, B
    
    def SaveCurrentFilteringInstantDataInObject():
        
        
        return

    ############################################################################
         # Normalize single value
    def DenormalizeValue(self, odometry):
        # Load train pos min
        file_path = 'C:\\Users\\simob\\Desktop\\Old_training\\Old_training\\02\\train_positions\\train_positions_min'
        mat = scipy.io.loadmat(file_path)
        dataMin = mat['dataMin']
        dataMin = dataMin.transpose()
        # Load train pos max
        file_path = 'C:\\Users\\simob\\Desktop\\Old_training\\Old_training\\02\\train_positions\\train_positions_max'
        mat = scipy.io.loadmat(file_path)
        dataMax = mat['dataMax']
        dataMax = dataMax.transpose()
        # Normalize pos
        norm_particle = np.zeros(odometry.shape)
        norm_factor_pos = dataMax - dataMin
        norm_particle[0:2] = np.multiply(odometry[0:2], norm_factor_pos)
        norm_particle[0:2] = norm_particle[0:2] + dataMin
        # Normalize vel
        norm_factor_vel_x = dataMax[0,0] - dataMin[0,0]
        norm_particle[2] = np.multiply(odometry[2], norm_factor_vel_x)
        norm_factor_vel_y = dataMax[1,0] - dataMin[1,0]
        norm_particle[3] = np.multiply(odometry[3], norm_factor_vel_y)

        return norm_particle    

       # Normalize particles
    def NormalizeParticles(self, odometry):
        # Load train pos min
        file_path = 'C:\\Users\\simob\\Desktop\\Old_training\\Old_training\\02\\train_positions\\train_positions_min'
        mat = scipy.io.loadmat(file_path)
        dataMin = mat['dataMin']
        dataMin = dataMin.transpose()
        # Load train pos max
        file_path = 'C:\\Users\\simob\\Desktop\\Old_training\\Old_training\\02\\train_positions\\train_positions_max'
        mat = scipy.io.loadmat(file_path)
        dataMax = mat['dataMax']
        dataMax = dataMax.transpose()
        # Normalize pos
        norm_particle = np.zeros(odometry.shape)
        norm_particle[0,:] = (odometry[0,:] - dataMin[0]) / (dataMax[0] - dataMin[0])
        norm_particle[1,:] = (odometry[1,:] - dataMin[1]) / (dataMax[1] - dataMin[1])
        # Normalize vel
        norm_particle[2,:] = odometry[2,:] / (dataMax[0] - dataMin[0])
        norm_particle[3,:] = odometry[3,:] / (dataMax[1] - dataMin[1])

        return norm_particle

    # Denormalize particles
    def DenormalizeParticles(self, odometry):
        # Load train pos min
        file_path = 'C:\\Users\\simob\\Desktop\\Old_training\\Old_training\\02\\train_positions\\train_positions_min'
        mat = scipy.io.loadmat(file_path)
        dataMin = mat['dataMin']
        dataMin = dataMin.transpose()
        # Load train pos max
        file_path = 'C:\\Users\\simob\\Desktop\\Old_training\\Old_training\\02\\train_positions\\train_positions_max'
        mat = scipy.io.loadmat(file_path)
        dataMax = mat['dataMax']
        dataMax = dataMax.transpose()
        # Denormalize pos
        denorm_particle = np.zeros(odometry.shape)
        denorm_particle[0,:] = odometry[0,:] * (dataMax[0]-dataMin[0]) + dataMin[0]
        denorm_particle[1,:] = odometry[1,:] * (dataMax[1]-dataMin[1]) + dataMin[1]
        # Denormalize vel
        denorm_particle[2,:] = odometry[2,:] * (dataMax[0]-dataMin[0])
        denorm_particle[3,:] = odometry[3,:] * (dataMax[1]-dataMin[1])

        return denorm_particle

    # Function to calculate the odometry value using IMU
    def CalculateOdometryFromIMU(self, orientation, acceleration, current_odometry,index, current_velocity):
        add_acceleration = True

        orientation = np.asarray(orientation)
        acceleration = np.asarray(acceleration)
        # Initial orientation and initial velocity
        initial_orientation = np.asarray(self.firstOrientationValue.clone())
        firstOdometryValue_numpy = np.asarray(self.firstOdometryValue)
        initial_vel = firstOdometryValue_numpy[2:,index]
        # Frame rate
        frame_rate = 30

        # From quaternions to rotations angles
        initial_yaw = quat2angle(initial_orientation[0,0], initial_orientation[0,1:4], output_unit='rad')[2]
        yaw = quat2angle(orientation[0,0], orientation[0,1:4], output_unit='rad')[2]

        # Current velocity
        # if self.timeInstant == 1:
        #     current_vel = current_odometry[2:4]
        # else:
        #     current_vel = np.transpose(current_params)[2:4,0]
        # current_vel = np.asarray(current_vel)
        current_vel = np.asarray(current_velocity)

        # Current position
        current_pos = current_odometry[0:2]
        current_pos = np.asarray(current_pos)

        # Absolute orientation (alpha_y)
        alpha_y = yaw - initial_yaw

        # Array for next odometry
        next_odometry = np.zeros((4,1))

        # Angle between current and initial velocity
        a_3d = np.append(current_vel, [0])
        b_3d = np.append(initial_vel, [0])
        c = np.cross(a_3d, b_3d)
        c = np.asarray(c)

        if c[2] < 0:
            theta = -np.arctan2(np.linalg.norm(c),np.dot(current_vel,initial_vel))
        else:
            theta = np.arctan2(np.linalg.norm(c),np.dot(current_vel,initial_vel))
        
        beta_i = theta

        blue_angle = beta_i + alpha_y

        # Define rotation matrix
        R = [[np.cos(blue_angle), -np.sin(blue_angle)],
            [np.sin(blue_angle), np.cos(blue_angle)]]

        R = np.asarray(R)

            #Calculate next velocity, next acceleration on xy and next position
        next_vel_without_acc = np.matmul(R, current_vel.transpose())
        next_vel_without_acc = next_vel_without_acc.transpose()

        if add_acceleration:
            # Calculate acceleration using angle btw current vel and axes
            angleBetweenVelAndAxes = np.angle(current_vel[0] + 1j * current_vel[1])
            # Project the acceleration on only the first component along the x and y axis
            next_acc = self.getAccelerationOnXYOnlyFirstComponent(acceleration, angleBetweenVelAndAxes)
        else:
            next_acc = [0, 0]
        
        next_acc = np.asarray(next_acc)

        # Divide by data rate
        next_acc_mod = 0.5*next_acc/(frame_rate * frame_rate)

        # Calculate next velocity
        next_acc = next_acc.transpose()
        next_vel = next_vel_without_acc + next_acc/frame_rate
        next_vel = next_vel.transpose()

        # Calculate position
        next_pos = current_pos[:].transpose() + next_vel_without_acc + next_acc_mod.transpose()
        next_pos = next_pos.transpose()

        # Define next odometry vector
        next_odometry = np.concatenate((next_pos,next_vel))

        return next_odometry

    # Function to project the acceleration on only the first component along the x and y axis
    def getAccelerationOnXYOnlyFirstComponent(self, acc_motion, angleBetweenVelAndAxes):
        # Calculate acceleration module 
        acc_motion = acc_motion.transpose()
        acc_module = acc_motion[0]

        # Acceleration's projection on x and y
        acc_x = acc_module * np.cos(angleBetweenVelAndAxes)
        acc_y = acc_module * np.sin(angleBetweenVelAndAxes)
        acc_xy = [acc_x, acc_y]

        return acc_xy

    ############################################################################
    
    # Code 110: Performing localization and anomaly detection given only the image data.
    # INPUTS:
    # - currentImagesBatch: image values of current batch
    # - currentParamsBatch: odometry values of current batch (just for comparisons)
    # - outputFolder: where to save the results
    # - type_of_weighting: in which way to combine the anomaly weights. 
    #      This value is given but not used. However, in the future it might be
    #      possible to change weighting type.
    # - knownStartingPoint: do we know the starting odometry point of the tracking?
    # - saveReconstructedImages: do you want to save the reconstructed images by the VAE?
    # - reconstructedImagesFolder: where to save the reconstructed images
    # - fastDistanceCalculation: do you want to use MSE instead of Bhattacharya
    #      distance for calculating the distance from the clusters. This increases
    #      the speed of the algorithm.
    # - percentageParticlesToReinitialize: how many particles to re-initialized, if
    #      restarting is used.
    def CombinedMJPFsVideoOdometrySingleMJPF(self, currentImagesBatch, currentParamsBatch, 
                                   outputFolder, type_of_weighting = 0, 
                                   knownStartingPoint = False, saveReconstructedImages = False,
                                   reconstructedImagesFolder = '', fastDistanceCalculation = False, 
                                   percentageParticlesToReinitialize = 0):

        print(self.anomaliesMeans.shape[0])
        #######################################################################
        # 1) ------ PARTICLE-INDEPENDENT CALCULATIONS
        # A) ENCODE AND DECODE
        # Performed calling the encoding and decoding method on the VAE object
        reconstructedImagesBatch, a_seq, a_mu, a_var = self.kvae.CallVAEEncoderAndDecoderOverBatchSequence(currentImagesBatch)
        # Also consider the reconstruction error and save it
        imageReconstructionAnomalies = F.mse_loss(reconstructedImagesBatch, currentImagesBatch,size_average=False)
        # Save the reconstructed images, if requested
        if saveReconstructedImages == True:            
            if not os.path.exists(reconstructedImagesFolder):
                os.makedirs(reconstructedImagesFolder)
            save_image(torch.squeeze(torch.squeeze(reconstructedImagesBatch)), 
                       reconstructedImagesFolder + '/%.4d.jpg' % self.timeInstant)            
        # This is just to take out one non-necessary dimension from a_mu
        a_mu_flattened   = a_mu[0, 0:1, :].clone()
        # B) Find distances from video clusters
        # Find distances from video clustering centers. This is done by passing the encoded values of the current batch 
        # and finding the distance from them to the centers of the clusters
        if type_of_weighting == 0 or self.timeInstant == 0:
            if fastDistanceCalculation == False:
                distancesFromVideo  = self.FindDistancesFromVideoClusters(a_mu)   
            else:
                distancesFromVideo  = self.clusterGraphVideo.FindDistancesFromVideoClustersAbsOfMeans(a_mu)  
            _distFromVideo      = distancesFromVideo[:, 0, :].clone()
            _distFromVideo      = _distFromVideo - torch.min(_distFromVideo) + 1
            # C) Calculating alpha from the cluster distances
            self.direct_alpha_from_video              = self.alphaDistProbVideo(_distFromVideo).clone()
            self.alpha_from_video_plus_sequencing     = self.direct_alpha_from_video.clone() + 1e-8
        #######################################################################
        # 2) ------ INITIALIZATIONS
        # IF WE DON'T KNOW THE ODOMETRY OF THE FIRST TIME INSTANT
        if self.timeInstant == 0 and knownStartingPoint == False:            
            # A) Initializing the particles of video MJPF based on alpha vector.
            self.mjpf_video.InitializeParticlesBasedOnGivenClusterProbabilities(torch.squeeze(self.alpha_from_video_plus_sequencing))
            # B) Initialize also odometry with same cluster assignments of video
            self.mjpf.InitializeParticlesBasedOnGivenClusterAssignments(self.mjpf_video.clusterAssignments)
        # IF WE KNOW THE ODOMETRY OF THE FIRST TIME INSTANT
        elif self.timeInstant == 0 and knownStartingPoint == True: 
            # A) Find the distances of the given odometry from the clusters
            distancesFromOdometryClustersFirstTimeInstant  = self.FindDistancesFromParamsClusters(currentParamsBatch)        
            _distFromOdometryClustersFirstTimeInstant      = distancesFromOdometryClustersFirstTimeInstant[:, 0, :].clone()
            _distFromOdometryClustersFirstTimeInstant      = _distFromOdometryClustersFirstTimeInstant - torch.min(_distFromOdometryClustersFirstTimeInstant) + 1
            # B) Calculating alpha from the cluster distances
            DirectAlphaFromOdometryClustersFirstTimeInstant = self.kvae.alphaDistProb(_distFromOdometryClustersFirstTimeInstant)            
            # C) Initializing the particles of video MJPF based on alpha vector
            self.mjpf_video.InitializeParticlesBasedOnGivenClusterProbabilities(
                torch.squeeze(DirectAlphaFromOdometryClustersFirstTimeInstant))
            # D) Initialize also odometry with same cluster assignments of video and with the
            #    GIVEN ODOMETRY
            self.mjpf.InitializeParticlesBasedOnGivenClusterAssignments(self.mjpf_video.clusterAssignments)
            self.mjpf.InitializeParticlesMeanGivenSingleValue(currentParamsBatch, self.mjpf.nodesCov[0,:,:]/1000)               
        #######################################################################
        # 3) ------ VIDEO UPDATE
        # A) Particles likelihoods, given the cluster to which they belong
        if type_of_weighting == 0:
            particlesLikelihoodGivenCluster = self.mjpf_video.ObtainParticleLikelihoodGivenProbabilitiesOfClusters(
                    self.alpha_from_video_plus_sequencing)
        # B) Perform video UPDATE using video clustering
        self.mjpf_video.PerformUpdateOfParticlesUsingMatrices(a_mu_flattened)   
        # 'a' state update just for debugging
        self.mjpf_video.ObtainAUpdateFromZUpdateForAllParticles()
        # C) Find evaluated odometry from video latent state values
        self.updatedValuesFromDMatrices           = self.mjpf_video.FindParticlesUsingDMatrices().clone()
        #######################################################################
        # 4) ------ ODOMETRY 'UPDATE'
        if self.timeInstant != 0 or knownStartingPoint == False:  
            # B) Update the odometry balancing prediction through transition matrices and 
            # prediction through matrices D and E.
            print('Before update: {}'.format(torch.mean(self.mjpf.particlesMeansUpdated.clone(), 1)))
            self.mjpf.UpdateParticlesMeansAndCovariancesGivenDifferentObservedStatesAndDifferentObservationCovariancesSingleMJPF(
                    self.updatedValuesFromDMatrices.clone(), self.clusterGraphVideo.nodesCovD)
            print('After update: {}'.format(torch.mean(self.mjpf.particlesMeansUpdated.clone(), 1)))
            # Saving the updated odometry value before resampling
        self.updatedOdometryValuesBeforeResampling = self.mjpf.particlesMeansUpdated.clone()
        #######################################################################
        # Calculate KLDA
        if self.timeInstant > 0:
            KLDA  = self.mjpf.CalculateOverallKLDA(self.alpha_from_video_plus_sequencing)
        else:
            KLDA  = torch.zeros(1)
        #######################################################################
        # 5) ------ VIDEO REWEIGHTING
        # A) Perform reweighting of particles based on alpha
        if type_of_weighting == 0: # using the cluster likelihoods + video anomaly
            anomalies_video         = self.mjpf_video.FindAStatesPredictionVsObservationAnomalies(a_mu) 
            self.anomalies_odometry = self.mjpf.FindStatePredictionVsExternalValuesAnomalyMSE(self.updatedValuesFromDMatrices.clone())
            self.mjpf_video.ReweightParticlesBasedOnAnomalyAndLikelihoods(anomalies_video,particlesLikelihoodGivenCluster)            
        else:            
            if type_of_weighting > 0:
                raise NonExistentTypeOfWeighting(type_of_weighting)
        #######################################################################
        # 6) ------ ANOMALIES CALCULATION     
        if self.timeInstant == 0:
            # Array where to save the time instants in which restarting is perfomed
            self.whenRestarted = []
            # Array to save the index of the anomaly causing the restart
            self.whyRestarted = []
            # percentage of particles that could be reinitialized
            self.numberOfParticlesToReinitialize = int(np.floor(percentageParticlesToReinitialize*self.mjpf.numberOfParticles/100)) 
        if self.timeInstant == 0 or self.reinitialized == True:           
            self.HandleAnomalyWindowingAfterRestart()  
        if self.timeAfterReinit > 0:
            anomalyVectorCurrentTimeInstant = []
            anomalyVectorCurrentTimeInstant.append(KLDA.item())
            anomalyVectorCurrentTimeInstant.append(torch.min(anomalies_video).item())
            anomalyVectorCurrentTimeInstant.append(imageReconstructionAnomalies.item())
            anomalyVectorCurrentTimeInstant.append(torch.min(particlesLikelihoodGivenCluster).item())
            diffsPreds = torch.linalg.norm(self.updatedValuesFromDMatrices - self.mjpf.particlesMeansPredicted, dim = 0)
            mean_diffsPreds = torch.mean(torch.mean(diffsPreds)).item()
            anomalyVectorCurrentTimeInstant.append(mean_diffsPreds)
            self.anomalyVectorCurrentTimeInstant = np.asarray(anomalyVectorCurrentTimeInstant)
        else:
            self.anomalyVectorCurrentTimeInstant = [0] * self.numberOfAnomalies
        #######################################################################
        # 7) ------ HANDLING ANOMALY WINDOWS
        # A) Handle the anomaly windows at the current time instant   
        if self.usingAnomalyThresholds == True:
            if self.timeAfterReinit > self.time_window:
                self.EliminateOldestValueFromAnomalyWindows()   
            if self.timeAfterReinit > 0:
                self.UpdateAnomalyWindows(anomalyVectorCurrentTimeInstant)
            # B) Calculate sum over the windows
            sumsOverAnomalyWindows = self.CalculateSumsOverAnomalyWindows()
            self.PrintSumsOverAnomalyWindows(sumsOverAnomalyWindows)
            # C) Check if it is necessary to restart
            #if self.timeAfterReinit > 0 and self.usingAnomalyThresholds == True:
                #self.CheckIfParticlesRestartingIsNecessary(sumsOverAnomalyWindows)
        #######################################################################
        # 8) ------ RESAMPLING AND RESTARTING         
        resamplingNecessary = self.mjpf_video.CheckIfResamplingIsNecessary()        
        # If resampling is necessary
        if resamplingNecessary or self.needReinit: # also resample when reinitialization is requested            
            print('RESAMPLE')   
            if self.needReinit: # Put the low resampling threshold (as it is a restart)
                self.mjpf.resamplingThreshold       = self.firstResampleThresh
                self.mjpf_video.resamplingThreshold = self.firstResampleThresh       
            else: # Put the high resampling threshhold
                self.mjpf.resamplingThreshold       = self.resampleThresh
                self.mjpf_video.resamplingThreshold = self.resampleThresh 
            #######################################################################
            # A) VIDEO RESAMPLING
            self.newIndicesForSwapping = self.mjpf_video.ResampleParticles()               
            #######################################################################
            # B) ODOMETRY RESAMPLING
            self.mjpf.ResampleParticlesGivenNewIndices(self.newIndicesForSwapping.copy())           
            #######################################################################
            # C) RESTARTING OF A SUBSET OF PARTICLES
            if self.needReinit == True:                     
                # Select the indices of particles with smallest weights
                # This is unnecessary if restarting is done after resampling, 
                # as the weights will be the same for all particles, but let's 
                # keep it in case of changes.
                selectedIndices = self.mjpf_video.SelectIndicesOfParticlesWithLowerWeights(
                        self.numberOfParticlesToReinitialize)
                # Restart the subset of particles
                for particleIndex in selectedIndices:    
                    particleIndex = int(particleIndex)
                    self.mjpf_video.InitializeParticleGivenClusterProbabilities(
                            torch.squeeze(self.alpha_from_video_plus_sequencing), particleIndex)
                    self.mjpf.InitializeParticleBasedOnGivenClusterAssignmentsAndIndex(
                            self.mjpf_video.clusterAssignments, particleIndex)                   
                self.reinitialized = True
                self.timeAfterReinit = -1
                self.whenRestarted.append(self.timeInstant)
                self.needReinit == False
            else:
                self.reinitialized = False            
        else:
            self.newIndicesForSwapping = np.zeros(self.mjpf.numberOfParticles)     
        if self.needReinit == False:
            self.indicesRestartedParticles = np.zeros(self.numberOfParticlesToReinitialize) 
        else:
            self.indicesRestartedParticles = np.asarray(selectedIndices)     
        #######################################################################
        # 9) ------ VIDEO PREDICTION
        # A) superstate prediction
        self.mjpf_video.PerformSuperstatesPrediction()
        self.mjpf.clusterAssignments = self.mjpf_video.clusterAssignments.copy()
        self.mjpf.timeInClusters     = self.mjpf_video.timeInClusters.clone()
        # B) state prediction        
        self.mjpf_video.PerformPredictionOfParticlesUsingMatrices()
        # a state prediction just for debugging
        self.mjpf_video.ObtainAPredictionFromZPredictionForAllParticles()        
        #######################################################################
        # 10) ------ ODOMETRY PREDICTION
        # A) state prediction
        self.mjpf.PredictParticlesMeansAndCovariances() 
        #######################################################################
        # 11) ------ Plus time
        self.timeInstant += 1
        self.timeAfterReinit += 1
        
        return

    ###########################################################################
    # Performing localization given the image data and IMU data

    def CombinedMJPFsVideoOdometrySingleMJPFWithIMU(self, currentImagesBatch, currentParamsBatch, currentAccBatch, currentOrientBatch,
                                   outputFolder, type_of_weighting = 0, 
                                   knownStartingPoint = False, saveReconstructedImages = False,
                                   reconstructedImagesFolder = '', fastDistanceCalculation = False, 
                                   percentageParticlesToReinitialize = 0):

        #######################################################################
        # 1) ------ PARTICLE-INDEPENDENT CALCULATIONS
        # A) ENCODE AND DECODE
        # Performed calling the encoding and decoding method on the VAE object
        reconstructedImagesBatch, a_seq, a_mu, a_var = self.kvae.CallVAEEncoderAndDecoderOverBatchSequence(currentImagesBatch)
        # Also consider the reconstruction error and save it
        imageReconstructionAnomalies = F.mse_loss(reconstructedImagesBatch, currentImagesBatch,size_average=False)
        # Save the reconstructed images, if requested
        if saveReconstructedImages == True:            
            if not os.path.exists(reconstructedImagesFolder):
                os.makedirs(reconstructedImagesFolder)
            save_image(torch.squeeze(torch.squeeze(reconstructedImagesBatch)), 
                       reconstructedImagesFolder + '/%.4d.jpg' % self.timeInstant)            
        # This is just to take out one non-necessary dimension from a_mu
        a_mu_flattened   = a_mu[0, 0:1, :].clone()
        # B) Find distances from video clusters
        # Find distances from video clustering centers. This is done by passing the encoded values of the current batch 
        # and finding the distance from them to the centers of the clusters
        if type_of_weighting == 0 or self.timeInstant == 0:
            if fastDistanceCalculation == False:
                distancesFromVideo  = self.FindDistancesFromVideoClusters(a_mu)   
            else:
                distancesFromVideo  = self.clusterGraphVideo.FindDistancesFromVideoClustersAbsOfMeans(a_mu)  
            _distFromVideo      = distancesFromVideo[:, 0, :].clone()
            _distFromVideo      = _distFromVideo - torch.min(_distFromVideo) + 1
            # C) Calculating alpha from the cluster distances
            self.direct_alpha_from_video              = self.alphaDistProbVideo(_distFromVideo).clone()
            self.alpha_from_video_plus_sequencing     = self.direct_alpha_from_video.clone() + 1e-8
        #######################################################################
        # 2) ------ INITIALIZATIONS
        # IF WE DON'T KNOW THE ODOMETRY OF THE FIRST TIME INSTANT
        if self.timeInstant == 0 and knownStartingPoint == False:            
            # A) Initializing the particles of video MJPF based on alpha vector.
            self.mjpf_video.InitializeParticlesBasedOnGivenClusterProbabilities(torch.squeeze(self.alpha_from_video_plus_sequencing))
            # B) Initialize also odometry with same cluster assignments of video
            self.mjpf.InitializeParticlesBasedOnGivenClusterAssignments(self.mjpf_video.clusterAssignments)
            # C) Initialize the first orientation value to calculate predictions with IMU
            self.firstOrientationValue = currentOrientBatch
        # IF WE KNOW THE ODOMETRY OF THE FIRST TIME INSTANT
        elif self.timeInstant == 0 and knownStartingPoint == True: 
            # A) Find the distances of the given odometry from the clusters
            distancesFromOdometryClustersFirstTimeInstant  = self.FindDistancesFromParamsClusters(currentParamsBatch)        
            _distFromOdometryClustersFirstTimeInstant      = distancesFromOdometryClustersFirstTimeInstant[:, 0, :].clone()
            _distFromOdometryClustersFirstTimeInstant      = _distFromOdometryClustersFirstTimeInstant - torch.min(_distFromOdometryClustersFirstTimeInstant) + 1
            # B) Calculating alpha from the cluster distances
            DirectAlphaFromOdometryClustersFirstTimeInstant = self.kvae.alphaDistProb(_distFromOdometryClustersFirstTimeInstant)            
            # C) Initializing the particles of video MJPF based on alpha vector
            self.mjpf_video.InitializeParticlesBasedOnGivenClusterProbabilities(
                torch.squeeze(DirectAlphaFromOdometryClustersFirstTimeInstant))
            # D) Initialize also odometry with same cluster assignments of video and with the
            #    GIVEN ODOMETRY
            self.mjpf.InitializeParticlesBasedOnGivenClusterAssignments(self.mjpf_video.clusterAssignments)
            self.mjpf.InitializeParticlesMeanGivenSingleValue(currentParamsBatch, self.mjpf.nodesCov[0,:,:]/1000)      
            # E) Initialize the first orientation value to calculate predictions with IMU
            self.firstOrientationValue = currentOrientBatch.clone() 
            firstOdometry = self.mjpf.particlesMeansUpdated.clone()  
            self.firstOdometryValue = self.DenormalizeParticles(firstOdometry)
            self.previousPosition = self.DenormalizeValue(np.transpose(currentParamsBatch.clone()))
        #######################################################################
        # 3) ------ VIDEO UPDATE
        # A) Particles likelihoods, given the cluster to which they belong
        if type_of_weighting == 0:
            particlesLikelihoodGivenCluster = self.mjpf_video.ObtainParticleLikelihoodGivenProbabilitiesOfClusters(
                    self.alpha_from_video_plus_sequencing)
        # B) Perform video UPDATE using video clustering
        self.mjpf_video.PerformUpdateOfParticlesUsingMatrices(a_mu_flattened)   
        # 'a' state update just for debugging
        self.mjpf_video.ObtainAUpdateFromZUpdateForAllParticles()
        # C) Find evaluated odometry from video latent state values
        self.updatedValuesFromDMatrices           = self.mjpf_video.FindParticlesUsingDMatrices().clone()
        #######################################################################
        # 4) ------ ODOMETRY 'UPDATE'
        if self.timeInstant != 0 or knownStartingPoint == False:  
            # B) Update the odometry balancing prediction through transition matrices and 
            # prediction through matrices D and E.
            if self.timeInstant == 1:
                # First odometry value to calculate predictions with IMU
                currentParamsDenorm = self.DenormalizeParticles(np.transpose(currentParamsBatch.clone()))
                current_vel = (currentParamsDenorm - self.previousPosition)[0:2,0]
                self.previousPosition = currentParamsDenorm
                for j in range(self.firstOdometryValue.shape[1]):
                    current_odometry = self.firstOdometryValue[:,j]
                    self.estimatedOdometryFromIMU[:,j] = self.CalculateOdometryFromIMU(currentOrientBatch, currentAccBatch, current_odometry, j, current_vel)[:,0]
            elif self.timeInstant != 0:
                # currentGT = torch.transpose(currentParamsBatch, 0, 1)
                # currentGT = currentGT.repeat(1, self.mjpf.numberOfParticles)
                # currentOdometryParticles = self.DenormalizeParticle(currentGT)
                currentOdometryParticles = self.DenormalizeParticles(self.mjpf.particlesMeansUpdated.clone())
                currentParamsDenorm = self.DenormalizeParticles(np.transpose(currentParamsBatch.clone()))
                current_vel = (currentParamsDenorm - self.previousPosition)[0:2,0]
                self.previousPosition = currentParamsDenorm
                for j in range(self.mjpf.numberOfParticles):
                    current_odometry = currentOdometryParticles[:,j]
                    self.estimatedOdometryFromIMU[:,j] = self.CalculateOdometryFromIMU(currentOrientBatch, currentAccBatch, current_odometry , j, current_vel)[:,0]
            
            self.estimatedOdometryFromIMU = self.NormalizeParticles(self.estimatedOdometryFromIMU)
            self.mjpf.UpdateParticlesMeansAndCovariancesGivenDifferentObservedStatesAndDifferentObservationCovariancesSingleMJPF(
                self.updatedValuesFromDMatrices.clone(), self.clusterGraphVideo.nodesCovD)

        # Saving the updated odometry value before resampling
        self.mjpf.particlesMeansUpdated[2:4,:] = self.mjpf.differenceOfMeansBetweenUpdates.clone()[0:2,:]
        self.updatedOdometryValuesBeforeResampling = self.mjpf.particlesMeansUpdated.clone() # 4x50 - 50 particelle
        #######################################################################
        # Calculate KLDA
        if self.timeInstant > 0:
            # KLDA  = self.mjpf.CalculateOverallKLDA(self.alpha_from_video_plus_sequencing)
            KLDA  = self.mjpf.CalculateKLDAForEachParticle(self.alpha_from_video_plus_sequencing)
        else:
            # KLDA  = torch.zeros(1)
            KLDA  = torch.zeros(self.mjpf.numberOfParticles)
        #######################################################################
        # 5) ------ VIDEO REWEIGHTING
        # A) Perform reweighting of particles based on alpha
        if type_of_weighting == 0: # using the cluster likelihoods + video anomaly
            anomalies_video         = self.mjpf_video.FindAStatesPredictionVsObservationAnomalies(a_mu) 
            self.anomalies_odometry = self.mjpf.FindStatePredictionVsExternalValuesAnomalyMSE(self.updatedValuesFromDMatrices.clone())
            self.mjpf_video.ReweightParticlesBasedOnAnomalyAndLikelihoods(anomalies_video,particlesLikelihoodGivenCluster)            
        else:            
            if type_of_weighting > 0:
                raise NonExistentTypeOfWeighting(type_of_weighting)
        #######################################################################
        # 6) ------ ANOMALIES CALCULATION     
        if self.timeInstant == 0:
            # Array where to save the time instants in which restarting is perfomed
            self.whenRestarted = []
            # Array to save the index of the anomaly causing the restart
            self.whyRestarted = []
            # percentage of particles that could be reinitialized
            self.numberOfParticlesToReinitialize = int(np.floor(percentageParticlesToReinitialize*self.mjpf.numberOfParticles/100)) 
        if self.timeInstant == 0 or self.reinitialized == True:           
            self.HandleAnomalyWindowingAfterRestart()  
        if self.timeAfterReinit > 0:
            anomalyVectorCurrentTimeInstant = []
            # anomalyVectorCurrentTimeInstant.append(KLDA.item())
            anomalyVectorCurrentTimeInstant.append(torch.min(KLDA).item())
            anomalyVectorCurrentTimeInstant.append(torch.min(anomalies_video).item())
            anomalyVectorCurrentTimeInstant.append(imageReconstructionAnomalies.item())
            anomalyVectorCurrentTimeInstant.append(torch.min(particlesLikelihoodGivenCluster).item())
            diffsPreds = torch.linalg.norm(self.updatedValuesFromDMatrices - self.mjpf.particlesMeansPredicted, dim = 0)
            mean_diffsPreds = torch.mean(torch.mean(diffsPreds)).item()
            print(mean_diffsPreds)
            anomalyVectorCurrentTimeInstant.append(mean_diffsPreds)
            self.anomalyVectorCurrentTimeInstant = np.asarray(anomalyVectorCurrentTimeInstant)
        else:
            self.anomalyVectorCurrentTimeInstant = [0] * self.numberOfAnomalies

        print(self.anomalyVectorCurrentTimeInstant)

        #######################################################################
        # 7) ------ IMU UPDATE
        # Find the weight related to video for each particle
        if self.timeInstant > 0:
            weights = []
            for i in range(self.anomaliesMeans.shape[0]):
                mean = self.anomaliesMeans[i]
                std = self.anomaliesStandardDeviations[i]
                current_anomaly = self.anomalyVectorCurrentTimeInstant[i]
                weight = self.CalculateWeightForIMU(mean, std, current_anomaly)
                weights.append(weight)
            # weights_all_anomalies = torch.zeros(self.numberOfAnomalies, self.mjpf.numberOfParticles)
            # mean = torch.zeros(self.numberOfAnomalies,1)
            # std = torch.zeros(self.numberOfAnomalies,1)
            # for j in range(self.numberOfAnomalies):
            #     mean[j] = self.anomaliesMeans[j]
            #     std[j] = self.anomaliesStandardDeviations[j]

            # for i in range(self.mjpf.numberOfParticles):
            #     weights_all_anomalies[0,i] = self.CalculateWeightForIMU(mean[0].item(), std[0].item(), KLDA[i].item())
            #     weights_all_anomalies[1,i] = self.CalculateWeightForIMU(mean[1].item(), std[1].item(), anomalies_video[i].item())
            #     weights_all_anomalies[2,i] = self.CalculateWeightForIMU(mean[2].item(), std[2].item(), imageReconstructionAnomalies.item())
            #     weights_all_anomalies[3,i] = self.CalculateWeightForIMU(mean[3].item(), std[3].item(), particlesLikelihoodGivenCluster[i].item())
            #     weights_all_anomalies[4,i] = self.CalculateWeightForIMU(mean[4].item(), std[4].item(), diffsPreds[i].item())

            # weights = torch.mean(weights_all_anomalies, 0)
            # print(weights.shape)
            # # self.weights_mat = np.matlib.repmat(weights, 4, 1)
            # self.weights_mat = weights.repeat(4, 1)

            print('IMU weights: {}'.format(weights))
            weights = np.asarray(weights)
            min_indexes = np.argsort(weights)[:4]
            self.weight_IMU = np.mean(weights[min_indexes])
            print(self.weight_IMU)
        
            # if self.timeInstant > 1539 and self.timeInstant < 1800:
            #     self.weight_IMU = 0
            #     print('Weight = 0')
            # else: 
            #     self.weight_IMU = 1

            # Compute odometry update with a weighted mean between IMU prediction and video prediction
            # self.updateWithIMU  = \
            #     torch.mul(self.updatedOdometryValuesBeforeResampling.clone(), self.weights_mat.clone()) +\
            #         torch.mul(torch.from_numpy(self.estimatedOdometryFromIMU), (1 - self.weights_mat.clone()))
            self.updateWithIMU  = \
                self.updatedOdometryValuesBeforeResampling.clone() * self.weight_IMU +\
                self.estimatedOdometryFromIMU * (1 - self.weight_IMU)

            self.updatedOdometryValuesBeforeResampling = self.updateWithIMU.clone()
            self.mjpf.particlesMeansUpdated = self.updateWithIMU.clone()
        
        #######################################################################
        # 8) ------ HANDLING ANOMALY WINDOWS
        # A) Handle the anomaly windows at the current time instant   
        if self.usingAnomalyThresholds == True:
            if self.timeAfterReinit > self.time_window:
                self.EliminateOldestValueFromAnomalyWindows()   
            if self.timeAfterReinit > 0:
                self.UpdateAnomalyWindows(anomalyVectorCurrentTimeInstant)
            # B) Calculate sum over the windows
            sumsOverAnomalyWindows = self.CalculateSumsOverAnomalyWindows()
            self.PrintSumsOverAnomalyWindows(sumsOverAnomalyWindows)
            # C) Check if it is necessary to restart
            #if self.timeAfterReinit > 0 and self.usingAnomalyThresholds == True:
                #self.CheckIfParticlesRestartingIsNecessary(sumsOverAnomalyWindows)    

        #######################################################################
        # 9) ------ RESAMPLING AND RESTARTING         
        resamplingNecessary = self.mjpf_video.CheckIfResamplingIsNecessary()        
        # If resampling is necessary
        if resamplingNecessary or self.needReinit: # also resample when reinitialization is requested            
            print('RESAMPLE')   
            if self.needReinit: # Put the low resampling threshold (as it is a restart)
                self.mjpf.resamplingThreshold       = self.firstResampleThresh
                self.mjpf_video.resamplingThreshold = self.firstResampleThresh       
            else: # Put the high resampling threshhold
                self.mjpf.resamplingThreshold       = self.resampleThresh
                self.mjpf_video.resamplingThreshold = self.resampleThresh 
            #######################################################################
            # A) VIDEO RESAMPLING
            self.newIndicesForSwapping = self.mjpf_video.ResampleParticles()               
            #######################################################################
            # B) ODOMETRY RESAMPLING
            self.mjpf.ResampleParticlesGivenNewIndices(self.newIndicesForSwapping.copy())           
            #######################################################################
            # C) RESTARTING OF A SUBSET OF PARTICLES
            if self.needReinit == True:                     
                # Select the indices of particles with smallest weights
                # This is unnecessary if restarting is done after resampling, 
                # as the weights will be the same for all particles, but let's 
                # keep it in case of changes.
                selectedIndices = self.mjpf_video.SelectIndicesOfParticlesWithLowerWeights(
                        self.numberOfParticlesToReinitialize)
                # Restart the subset of particles
                for particleIndex in selectedIndices:    
                    particleIndex = int(particleIndex)
                    self.mjpf_video.InitializeParticleGivenClusterProbabilities(
                            torch.squeeze(self.alpha_from_video_plus_sequencing), particleIndex)
                    self.mjpf.InitializeParticleBasedOnGivenClusterAssignmentsAndIndex(
                            self.mjpf_video.clusterAssignments, particleIndex)                   
                self.reinitialized = True
                self.timeAfterReinit = -1
                self.whenRestarted.append(self.timeInstant)
                self.needReinit == False
            else:
                self.reinitialized = False            
        else:
            self.newIndicesForSwapping = np.zeros(self.mjpf.numberOfParticles)     
        if self.needReinit == False:
            self.indicesRestartedParticles = np.zeros(self.numberOfParticlesToReinitialize) 
        else:
            self.indicesRestartedParticles = np.asarray(selectedIndices)     

        #######################################################################
        # 9) ------ VIDEO PREDICTION
        # A) superstate prediction
        self.mjpf_video.PerformSuperstatesPrediction()
        self.mjpf.clusterAssignments = self.mjpf_video.clusterAssignments.copy()
        self.mjpf.timeInClusters     = self.mjpf_video.timeInClusters.clone()
        # B) state prediction        
        self.mjpf_video.PerformPredictionOfParticlesUsingMatrices()
        # a state prediction just for debugging
        self.mjpf_video.ObtainAPredictionFromZPredictionForAllParticles()       

        #######################################################################
        # 10) ------ ODOMETRY PREDICTION
        # A) state prediction
        self.mjpf.PredictParticlesMeansAndCovariances() 
        
        #######################################################################
        # 11) ------ Plus time
        self.timeInstant += 1
        self.timeAfterReinit += 1
        
        return

    def CalculateWeightForIMU(self, anomaliesMean, anomaliesStd, current_anomaly):
        weight = scipy.stats.norm(anomaliesMean, anomaliesStd).pdf(current_anomaly)
        x = np.linspace(anomaliesMean - 3*anomaliesStd, anomaliesMean + 3*anomaliesStd, 1000)
        gaussian_distribution = scipy.stats.norm(anomaliesMean, anomaliesStd).pdf(x)
        minValue = np.min(gaussian_distribution)
        maxValue = np.max(gaussian_distribution)
        normalized_weight = np.absolute((weight - minValue) / (maxValue - minValue))

        return normalized_weight
    
    ###########################################################################
    # Functions for plotting prediction and updates at image level against
    # the real image.
    
    def GetImagePredictionFromZPredictedSingleParticle(self, particleIndex):
        
        a_state_predicted = self.mjpf_video.ObtainAPredictionFromZPredictionGivenParticleIndex(particleIndex)
        image_predicted   = self.kvae.baseVAE.Decode(a_state_predicted)        
        return image_predicted
    
    def GetImageUpdateFromZUpdatedSingleParticle(self, particleIndex):
        
        a_state_updated   = self.mjpf_video.ObtainAUpdateFromZUpdateGivenParticleIndex(particleIndex)
        image_updated     = self.kvae.baseVAE.Decode(a_state_updated)        
        return image_updated
    
    def GetImageUpdateAndPredictionFromZStatesSingleParticle(self, particleIndex):
        
        # From state z to state a, for prediction and update
        image_updated     = self.GetImageUpdateFromZUpdatedSingleParticle(particleIndex)
        image_predicted   = self.GetImagePredictionFromZPredictedSingleParticle(particleIndex)        
        return image_updated, image_predicted
    
    def PrintRealImageVsUpdatedAndPredictedSingleParticle(self, currentImage, particleIndex, fileName):
        
        image_updated, image_predicted = self.GetImageUpdateAndPredictionFromZStatesSingleParticle(particleIndex)
        
        # Squeeze the images to eliminate dimensions of size 1
        squeezedCurrentImage    = torch.squeeze(currentImage)
        squeezedUpdatedImage    = torch.squeeze(image_updated)
        squeezedPredictedImage  = torch.squeeze(image_predicted)
        dimensionsSqueezedImage = squeezedCurrentImage.ndim
        
        # Put together the real, updated and predicted images. 
        # If the images are grayscale, simply concatenate them.
        # If they are RGB, a dimension must be created on which to concatenate.
        if dimensionsSqueezedImage == 2:   # grayscale image
            imagesToPrint = torch.cat([squeezedCurrentImage, 
                                       squeezedUpdatedImage, 
                                       squeezedPredictedImage])
        elif dimensionsSqueezedImage == 3: # colored image
            imagesToPrint = torch.cat([torch.unsqueeze(squeezedCurrentImage,0), 
                                       torch.unsqueeze(squeezedUpdatedImage,0), 
                                       torch.unsqueeze(squeezedPredictedImage,0)])
        
        save_image(imagesToPrint, fileName)
        return
    
###############################################################################
# Tests classes for checking secondary parts of the code.
        
class TestsKvaeOdometryFromVideo(unittest.TestCase):
    
    # Initializing a dummy KVAE_odometry_from_video object, that does not actually
    # contain any of its internal objects (kvae, mjpf)
    def _InitializeDummyKVAEOFV(self, kvae = None, clusterGraphVideo = None, clusterGraphVideoZ = None, 
                               clusterGraphParams = None, mjpf = None, mjpf_video = None, 
                               skew_video = 1, anomaliesMeans = None, anomaliesStandardDeviations = None, 
                               time_window = 20, time_enough_ratio = 0.6, time_wait_ratio = 0.25, 
                               stdTimes = 2.5, usingAnomalyThresholds = False):
        
        self.kvaeOfV = KVAE_odometry_from_video(kvae = kvae, clusterGraphVideo = clusterGraphVideo, 
           clusterGraphVideoZ = clusterGraphVideoZ, clusterGraphParams = clusterGraphParams, mjpf = mjpf, 
           mjpf_video = mjpf_video, skew_video = skew_video, anomaliesMeans = anomaliesMeans, 
           anomaliesStandardDeviations = anomaliesStandardDeviations, time_window = time_window, 
           time_enough_ratio = time_enough_ratio, time_wait_ratio = time_wait_ratio, stdTimes = stdTimes,
           usingAnomalyThresholds = usingAnomalyThresholds)        
        return
    
    def CheckTimeEnoughCalculation(self):
        
        time_window = 20
        time_enough_ratio = 0.75 # 15
        
        self._InitializeDummyKVAEOFV(time_window = time_window, time_enough_ratio = time_enough_ratio)        
        self.assertTrue(self.kvaeOfV.time_enough == 15) 
        return
    
    def CheckTimeWaitCalculation(self):
        
        time_window = 20
        time_wait_ratio = 0.25 # 5
        
        self._InitializeDummyKVAEOFV(time_window = time_window, time_wait_ratio = time_wait_ratio)        
        self.assertTrue(self.kvaeOfV.time_wait == 5) 
        return
    
    def _DefineDummyAnomalyThresholdsBuildingComponenets(self):
        
        anomaliesMeans = np.array([1,2,3,4])
        anomaliesStandardDeviations = np.array([0.1,0.2,0.3,0.4])
        stdTimes = 2
        
        return anomaliesMeans, anomaliesStandardDeviations, stdTimes
    
    def CheckAnomalyThresholdsCalculation(self):
        
        anomaliesMeans, anomaliesStandardDeviations, stdTimes = \
           self._DefineDummyAnomalyThresholdsBuildingComponenets()
        self._InitializeDummyKVAEOFV(anomaliesMeans = anomaliesMeans, 
                                    anomaliesStandardDeviations = anomaliesStandardDeviations, 
                                    stdTimes = stdTimes, usingAnomalyThresholds = True)
        # Check
        expectedAnomaliesThresholds = np.array([1.2,2.4,3.6,4.8])
        self.assertTrue((expectedAnomaliesThresholds == self.kvaeOfV.anomalyThresholds).all())        
        return
    
    def _CheckLengthOfInitializedAnomalyWindowsIsCorrect(self,numberOfAnomalies):
        
        self.assertTrue(numberOfAnomalies == len(self.kvaeOfV.anomalyWindows))         
        return
    
    def _CheckThatInitializedAnomalyWindowsAreEmpty(self):
        
        numberOfAnomalies = len(self.kvaeOfV.anomalyWindows)
        for indexOfAnomaly in range(numberOfAnomalies):
            currentWindow = self.kvaeOfV.anomalyWindows[indexOfAnomaly]
            currentWindowLength = len(currentWindow)
            self.assertTrue(currentWindowLength == 0) 
        return
    
    def CheckAnomalyWindowsInitializationAtStart(self):
        
        # Dummy initialization
        self._InitializeDummyKVAEOFV()      
        # Check
        numberOfAnomalies = self.kvaeOfV.numberOfAnomalies
        self._CheckLengthOfInitializedAnomalyWindowsIsCorrect(numberOfAnomalies)
        self._CheckThatInitializedAnomalyWindowsAreEmpty()
        return
    
    def CheckAnomalyWindowsAfterSecondInitialization(self):
        
        # Dummy initialization
        self._InitializeDummyKVAEOFV()  
        # Reinitialize the anomaly windows
        self.kvaeOfV.InitializeAnomalyWindows()
        # Check
        numberOfAnomalies = self.kvaeOfV.numberOfAnomalies
        self._CheckLengthOfInitializedAnomalyWindowsIsCorrect(numberOfAnomalies)
        self._CheckThatInitializedAnomalyWindowsAreEmpty()
        return
    
    def CheckEliminationOfOldestValueFromSingleAnomalyWindow(self):
        
        # Dummy initialization
        self._InitializeDummyKVAEOFV()  
        # Add values to one of the anomaly windows
        indexOfAnomaly = 0
        self.kvaeOfV.anomalyWindows[indexOfAnomaly].append(0)
        self.kvaeOfV.anomalyWindows[indexOfAnomaly].append(1)
        self.kvaeOfV.anomalyWindows[indexOfAnomaly].append(1)
        # Eliminate the oldest value
        self.kvaeOfV.EliminateOldestValueFromSingleAnomalyWindow(indexOfAnomaly)
        # Check
        expectedAnomalyWindow = np.array([1,1])
        obtainedAnomalyWindow = self.kvaeOfV.anomalyWindows[indexOfAnomaly]
        self.assertTrue((expectedAnomalyWindow == obtainedAnomalyWindow).all())        
        return
    
    def CheckUpdateOfSingleAnomalyWindowWhenOverThreshold(self):
        
        # Dummy initialization
        anomaliesMeans, anomaliesStandardDeviations, stdTimes = \
           self._DefineDummyAnomalyThresholdsBuildingComponenets()
        self._InitializeDummyKVAEOFV(anomaliesMeans = anomaliesMeans, 
                                    anomaliesStandardDeviations = anomaliesStandardDeviations, 
                                    stdTimes = stdTimes, usingAnomalyThresholds = True)
        indexOfAnomaly = 0
        # An anomaly over the threshold (1.2)
        anomalyValue = 2000 
        # Update
        self.kvaeOfV.UpdateSingleAnomalyWindow(indexOfAnomaly, anomalyValue)
        # Check
        expectedAnomalyWindow = np.array([1])
        obtainedAnomalyWindow = self.kvaeOfV.anomalyWindows[indexOfAnomaly]
        self.assertTrue((expectedAnomalyWindow == obtainedAnomalyWindow).all())  
        return
    
    def CheckUpdateOfSingleAnomalyWindowWhenBelowThreshold(self):
        
        # Dummy initialization
        anomaliesMeans, anomaliesStandardDeviations, stdTimes = \
           self._DefineDummyAnomalyThresholdsBuildingComponenets()
        self._InitializeDummyKVAEOFV(anomaliesMeans = anomaliesMeans, 
                                    anomaliesStandardDeviations = anomaliesStandardDeviations, 
                                    stdTimes = stdTimes, usingAnomalyThresholds = True)
        indexOfAnomaly = 0
        # An anomaly below the threshold (1.2)
        anomalyValue = 0.00000001
        # Update
        self.kvaeOfV.UpdateSingleAnomalyWindow(indexOfAnomaly, anomalyValue)
        # Check
        expectedAnomalyWindow = np.array([0])
        obtainedAnomalyWindow = self.kvaeOfV.anomalyWindows[indexOfAnomaly]
        self.assertTrue((expectedAnomalyWindow == obtainedAnomalyWindow).all())  
        return
    
    def CheckSumOverSingleAnomalyWindow(self):
        
        # Dummy initialization
        anomaliesMeans, anomaliesStandardDeviations, stdTimes = \
           self._DefineDummyAnomalyThresholdsBuildingComponenets()
        self._InitializeDummyKVAEOFV(anomaliesMeans = anomaliesMeans, 
                                    anomaliesStandardDeviations = anomaliesStandardDeviations, 
                                    stdTimes = stdTimes, usingAnomalyThresholds = True)
        indexOfAnomaly = 0
        # An anomaly below the threshold (1.2)
        anomalyValueBelow = 0.00000001
        # An anomaly over the threshold (1.2)
        anomalyValueAbove = 2000 
        # Update
        self.kvaeOfV.UpdateSingleAnomalyWindow(indexOfAnomaly, anomalyValueAbove)
        self.kvaeOfV.UpdateSingleAnomalyWindow(indexOfAnomaly, anomalyValueBelow)
        self.kvaeOfV.UpdateSingleAnomalyWindow(indexOfAnomaly, anomalyValueAbove)
        self.kvaeOfV.UpdateSingleAnomalyWindow(indexOfAnomaly, anomalyValueAbove)
        # Calculate sum
        sumOverSingleAnomalyWindow = self.kvaeOfV.CalculateSumOverSingleAnomalyWindow(indexOfAnomaly)
        # Check
        expectedSumOverSingleAnomalyWindow = 3
        self.assertTrue(sumOverSingleAnomalyWindow,expectedSumOverSingleAnomalyWindow)  
        return
    
    def _FindSumOverAnomalyWindowsWhenAnomalyAtDefinedIndexIsAboveThreshold(self, indexOfAnomaly):
        
        time_window = 10
        time_enough_ratio = 0.6 # 6
        time_wait_ratio = 0.2 # 2
        # 5 anomalies, at index 2 is a waiting anomaly
        self._InitializeDummyKVAEOFV(time_window = time_window, time_enough_ratio = time_enough_ratio,
                                     time_wait_ratio = time_wait_ratio)
        # Manually define anomaly windows
        self.kvaeOfV.anomalyWindows[indexOfAnomaly] = [1,1,0,1,1,1,1,1]
        # Calculate sum
        sumOverAnomalyWindows = self.kvaeOfV.CalculateSumsOverAnomalyWindows()        
        return sumOverAnomalyWindows
    
    def CheckThatRestartingIsActivatedWhenStandardAnomalyIsAboveThreshold(self):
        
        indexOfAnomaly = 0
        
        sumOverAnomalyWindows = self._FindSumOverAnomalyWindowsWhenAnomalyAtDefinedIndexIsAboveThreshold(indexOfAnomaly)
        # Restart needed?
        self.kvaeOfV.CheckIfParticlesRestartingIsNecessary(sumOverAnomalyWindows)
        # Check 
        self.assertTrue(self.kvaeOfV.needReinit)
        self.assertFalse(self.kvaeOfV.needReinitAfterItIsGoodAgain)        
        return
    
    def CheckThatRestartingIsSetInWaitingAfterWaitingAnomalyIsAboveThreshold(self):
        
        indexOfAnomaly = 2
        
        sumOverAnomalyWindows = self._FindSumOverAnomalyWindowsWhenAnomalyAtDefinedIndexIsAboveThreshold(indexOfAnomaly)
        # Restart needed?
        self.kvaeOfV.CheckIfParticlesRestartingIsNecessary(sumOverAnomalyWindows)
        # Check 
        self.assertFalse(self.kvaeOfV.needReinit)
        self.assertTrue(self.kvaeOfV.needReinitAfterItIsGoodAgain)        
        return
    
    def CheckThatRestartingIsFiredAfterWaitingAnomalyReturnedNormal(self):
        
        indexOfAnomaly = 2
        
        time_window = 10
        time_enough_ratio = 0.6 # 6
        time_wait_ratio = 0.2 # 2
        # 5 anomalies, at index 2 is a waiting anomaly
        self._InitializeDummyKVAEOFV(time_window = time_window, time_enough_ratio = time_enough_ratio,
                                     time_wait_ratio = time_wait_ratio)
        self.kvaeOfV.needReinitAfterItIsGoodAgain = True
        # Manually define anomaly windows
        self.kvaeOfV.anomalyWindows[indexOfAnomaly] = [1,1,0,0,0,1,0,0]
        # Calculate sum
        sumOverAnomalyWindows = self.kvaeOfV.CalculateSumsOverAnomalyWindows()
        # Restart needed?
        self.kvaeOfV.CheckIfParticlesRestartingIsNecessary(sumOverAnomalyWindows)
        # Check 
        self.assertTrue(self.kvaeOfV.needReinit)
        self.assertFalse(self.kvaeOfV.needReinitAfterItIsGoodAgain)       
        return
    
    @staticmethod
    def PerformAllTests():
        
        TestKvaeOfV = TestsKvaeOdometryFromVideo()
        TestKvaeOfV.CheckTimeEnoughCalculation()
        TestKvaeOfV.CheckTimeWaitCalculation()
        TestKvaeOfV.CheckAnomalyThresholdsCalculation() 
        TestKvaeOfV.CheckAnomalyWindowsInitializationAtStart()
        TestKvaeOfV.CheckAnomalyWindowsAfterSecondInitialization()
        TestKvaeOfV.CheckEliminationOfOldestValueFromSingleAnomalyWindow()
        TestKvaeOfV.CheckUpdateOfSingleAnomalyWindowWhenOverThreshold()
        TestKvaeOfV.CheckUpdateOfSingleAnomalyWindowWhenBelowThreshold()
        TestKvaeOfV.CheckSumOverSingleAnomalyWindow()
        TestKvaeOfV.CheckThatRestartingIsActivatedWhenStandardAnomalyIsAboveThreshold()
        TestKvaeOfV.CheckThatRestartingIsSetInWaitingAfterWaitingAnomalyIsAboveThreshold()
        TestKvaeOfV.CheckThatRestartingIsFiredAfterWaitingAnomalyReturnedNormal()
        print('All tests have been successfully performed')
        return

def main():
    TestsKvaeOdometryFromVideo.PerformAllTests()
    
main() 
    
