# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:06:54 2022

@author: asus
"""

import os
import numpy as np
import pickle
import mat4py

from KVAE_models_and_utils_codes import ImagesHolder as IH

###############################################################################
# PROTOCOL of pickle
protocol = 4
def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        if protocol == 2:
            pickle.dump(di_, f, protocol = 2)
        if protocol == 4:
            pickle.dump(di_, f, protocol = 4)
        else:
            pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di
###############################################################################
# Extraction without IMU data
def ExtractDataForKVAE(config, path_to_GSs, path_to_GSs_cells, path_to_images_folder, 
                       path_to_inputs_to_final_model, path_to_inputs_to_final_model_full, check_mode):
    
    sequenceLength = config['sequence_length']
    
    ###########################################################################
    # Check whether the data is present
    if (not os.path.exists(path_to_GSs) # GS folder not exists?
        or not os.path.isdir(path_to_images_folder) # image folder not exists?
        or not os.listdir(path_to_images_folder)): # image subfolders don't exists?
        print ('Data in either {} or {} does not exist'.format(path_to_images_folder, path_to_GSs))
        # Continue if it is not present...
        return
    ###########################################################################
    # Loading the odometry data
    print('Loading odometry data')
    odometry = mat4py.loadmat(path_to_GSs)['data'] # this is a list of lists 
    odometry = np.asarray(odometry)
    odometry_cells = mat4py.loadmat(path_to_GSs_cells)['data'] # this is a list of lists 
    dimensionOfOdometryGSs = odometry.shape[1]
    ###########################################################################
    # Find number of trajectories and the starting points of each trajectory
    startingPoints = []
    startingPoints.append(0)
    if len(odometry_cells) == len(odometry): 
        # If there is only one trajectory, 'odometry_cells' is not loaded as
        # an array of trajectories, but as an array of single datapoints, so
        # its length corresponds to the one of 'odometry'.
        numberOfTrajectories = 1
    else:
        numberOfTrajectories = len(odometry_cells)
        # Find starting points
        sumUntilNow = 0
        for i in range(numberOfTrajectories-1):
            sumUntilNow += len(odometry_cells[i][0])
            startingPoints.append(sumUntilNow)
    ###########################################################################
    # Loading the image data
    print('Loading image data')
    dataImages = IH.ImagesHolder(path_to_images_folder, dimension_x = config['dimension_x'] , 
                                   dimension_y = config['dimension_y'], image_channels = config['image_channels'])
    # Length of data and number of channels
    if dataImages.images.ndim == 3:
        numberOfPoints   = dataImages.images.shape[0]
        numberOfChannels = 1
    else:
        numberOfPoints   = dataImages.images.shape[1]
        numberOfChannels = dataImages.images.shape[0]
    ###########################################################################
    # Preparing data with division in batches
    numberOfSequences = int(np.floor(numberOfPoints/sequenceLength))
    if numberOfChannels == 1:
        data     = np.zeros((numberOfSequences, sequenceLength, config['dimension_x'], config['dimension_y']))
    else:
        data     = np.zeros((numberOfChannels, numberOfSequences, sequenceLength, config['dimension_x'], config['dimension_y']))
    odom     = np.zeros((numberOfSequences, sequenceLength, 2))
    params   = np.zeros((numberOfSequences, sequenceLength, dimensionOfOdometryGSs))
    controls = np.zeros((numberOfSequences, sequenceLength, 1))
    ###########################################################################
    # MAIN loop of DATA EXTRACTION
    currBatch_idx = 0 # 
    for idx in range(numberOfPoints):
        print (str(idx) + " out of " + str(numberOfPoints))
        # In which sequence are we?
        sequence_number = int(np.floor(idx /sequenceLength));
        if sequence_number == numberOfSequences:
            continue
        # Printing to check
        if check_mode == True:
            print(sequence_number)
            print(currBatch_idx)
        # Insert
        if numberOfChannels == 1:
            data[sequence_number, currBatch_idx, :, :]   = dataImages.images[idx, :, :]
        else:
            data[:,sequence_number, currBatch_idx, :, :] = dataImages.images[:, idx, :, :]
        params[sequence_number, currBatch_idx, :]  = odometry[idx, :] 
        odom[sequence_number, currBatch_idx, :]    = odometry[idx, 0:2] 
        # Increase index in batch
        currBatch_idx += 1
        # Have we finished the current batch and can we move to the next one?
        if currBatch_idx >= sequenceLength:
            currBatch_idx = 0
    ###########################################################################
    final_data = {'images':data, 'odometry': odom ,'controls':controls, 'params' : params, 
                  'timesteps':sequenceLength, 'sequences':numberOfSequences, 
                  'd1': config['dimension_x'], 'd2': config['dimension_y'], 
                  'startingPoints': startingPoints}
    ###########################################################################
    # Saving
    save_dict(final_data, path_to_inputs_to_final_model_full)
    save_dict(final_data, path_to_inputs_to_final_model)
    
    del dataImages, odometry
    del data, odom, params, controls
    del final_data


# Extraction including IMU data
def ExtractDataForKVAE(config, path_to_GSs, path_to_GSs_cells, path_to_images_folder, path_to_acc, path_to_angularVel,
                       path_to_inputs_to_final_model, path_to_inputs_to_final_model_full, check_mode):
    
    sequenceLength = config['sequence_length']
    
    ###########################################################################
    # Check whether the data is present
    if (not os.path.exists(path_to_GSs) # GS folder not exists?
        or not os.path.isdir(path_to_images_folder) # image folder not exists?
        or not os.path.exists(path_to_acc) # acceleration folder not exists?
        or not os.path.exists(path_to_angularVel) # angular velocity folder not exists?
        or not os.listdir(path_to_images_folder)): # image subfolders don't exists?
        print ('Data in {}, {}, {} or {} does not exist'.format(path_to_images_folder, path_to_GSs, path_to_acc, path_to_angularVel))
        # Continue if it is not present...
        return
    ###########################################################################
    # Loading IMU (acceleration and angular velocity) data
    print('Loading IMU (acceleration and angular velocity) data')
    acceleration = mat4py.loadmat(path_to_acc)['linearAccelerationSynch']
    acceleration = np.asarray(acceleration)
    angular_velocity = mat4py.loadmat(path_to_angularVel)['angularVelocitySynch']
    angular_velocity = np.asarray(angular_velocity)
    # Loading the odometry data
    print('Loading odometry data')
    odometry = mat4py.loadmat(path_to_GSs)['data'] # this is a list of lists 
    odometry = np.asarray(odometry)
    odometry_cells = mat4py.loadmat(path_to_GSs_cells)['data'] # this is a list of lists 
    dimensionOfOdometryGSs = odometry.shape[1]
    ###########################################################################
    # Find number of trajectories and the starting points of each trajectory
    startingPoints = []
    startingPoints.append(0)
    if len(odometry_cells) == len(odometry): 
        # If there is only one trajectory, 'odometry_cells' is not loaded as
        # an array of trajectories, but as an array of single datapoints, so
        # its length corresponds to the one of 'odometry'.
        numberOfTrajectories = 1
    else:
        numberOfTrajectories = len(odometry_cells)
        # Find starting points
        sumUntilNow = 0
        for i in range(numberOfTrajectories-1):
            sumUntilNow += len(odometry_cells[i][0])
            startingPoints.append(sumUntilNow)
    ###########################################################################
    # Loading the image data
    print('Loading image data')
    dataImages = IH.ImagesHolder(path_to_images_folder, dimension_x = config['dimension_x'] , 
                                   dimension_y = config['dimension_y'], image_channels = config['image_channels'])
    # Length of data and number of channels
    if dataImages.images.ndim == 3:
        numberOfPoints   = dataImages.images.shape[0]
        numberOfChannels = 1
    else:
        numberOfPoints   = dataImages.images.shape[1]
        numberOfChannels = dataImages.images.shape[0]
    ###########################################################################
    # Preparing data with division in batches
    numberOfSequences = int(np.floor(numberOfPoints/sequenceLength))
    if numberOfChannels == 1:
        data     = np.zeros((numberOfSequences, sequenceLength, config['dimension_x'], config['dimension_y']))
    else:
        data     = np.zeros((numberOfChannels, numberOfSequences, sequenceLength, config['dimension_x'], config['dimension_y']))
    odom     = np.zeros((numberOfSequences, sequenceLength, 2))
    acc     = np.zeros((numberOfSequences, sequenceLength, 2))
    ang_vel     = np.zeros((numberOfSequences, sequenceLength, 3))
    params   = np.zeros((numberOfSequences, sequenceLength, dimensionOfOdometryGSs))
    controls = np.zeros((numberOfSequences, sequenceLength, 1))
    ###########################################################################
    # MAIN loop of DATA EXTRACTION
    currBatch_idx = 0 # 
    for idx in range(numberOfPoints):
        print (str(idx) + " out of " + str(numberOfPoints))
        # In which sequence are we?
        sequence_number = int(np.floor(idx /sequenceLength))
        if sequence_number == numberOfSequences:
            continue
        # Printing to check
        if check_mode == True:
            print(sequence_number)
            print(currBatch_idx)
        # Insert
        if numberOfChannels == 1:
            data[sequence_number, currBatch_idx, :, :]   = dataImages.images[idx, :, :]
        else:
            data[:,sequence_number, currBatch_idx, :, :] = dataImages.images[:, idx, :, :]
        params[sequence_number, currBatch_idx, :]  = odometry[idx, :] 
        odom[sequence_number, currBatch_idx, :]    = odometry[idx, 0:2] 
        acc[sequence_number, currBatch_idx, :]     = acceleration[idx, 0:2]
        ang_vel[sequence_number, currBatch_idx, :]     = angular_velocity[idx, :]
        # Increase index in batch
        currBatch_idx += 1
        # Have we finished the current batch and can we move to the next one?
        if currBatch_idx >= sequenceLength:
            currBatch_idx = 0
    ###########################################################################
    final_data = {'images':data, 'odometry': odom ,'controls':controls, 'params' : params,
                  'acceleration': acc, 'angular_velocity': ang_vel, 
                  'timesteps':sequenceLength, 'sequences':numberOfSequences, 
                  'd1': config['dimension_x'], 'd2': config['dimension_y'], 
                  'startingPoints': startingPoints}
    ###########################################################################
    # Saving
    save_dict(final_data, path_to_inputs_to_final_model_full)
    save_dict(final_data, path_to_inputs_to_final_model)
    
    del dataImages, odometry
    del data, odom, params, controls
    del final_data
