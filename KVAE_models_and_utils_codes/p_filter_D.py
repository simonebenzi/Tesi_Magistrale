# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:39:07 2021

@author: Abrham
"""

import torch
import torch.nn as nn

from KVAE_models_and_utils_codes import p_filter

from ConfigurationFiles import Config_GPU as ConfigGPU

###############################################################################
# ---------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)

###############################################################################
# ------------------          Kalman Filter        --------------------------
###############################################################################

class KalmanFilter_D(p_filter.KalmanFilter):

    def __init__(self, dim_z, dim_y, num_clusters, sequence_length, init_kf_matrices,
                 init_cov, batch_size, noise_transition, noise_emission,
                 clusteringDimension, nodesMean):
        
        p_filter.KalmanFilter.__init__(self, dim_z, dim_y, num_clusters, sequence_length, init_kf_matrices, \
                 init_cov, batch_size, noise_transition, noise_emission)
        
        self.clusteringDimension = clusteringDimension        
        # Matrix for parameters from state z
        #self.D = nn.Parameter(init_kf_matrices*torch.randn(self.num_clusters, self.clusteringDimension, self.dim_z).to(device), requires_grad=True)
        self.D = nn.Parameter(torch.zeros(self.num_clusters, self.clusteringDimension, self.dim_z).to(device), requires_grad=True)
        # This is a constant to add together to D
        self.E = nn.Parameter(torch.zeros(self.num_clusters, self.clusteringDimension).to(device), requires_grad=True)
        
        self.nodesMean = nodesMean.to(device)

        return
    
    ###########################################################################
    # Extracting the D and E values for the cluster where alpha is maximum.
    
    def extract_D_of_highest_alpha(self, alpha):
        alpha                 = torch.squeeze(alpha)
        clusterOfHighestAlpha = torch.argmax(alpha)
        return self.extract_D_of_cluster_index(clusterOfHighestAlpha)
    
    def extract_E_of_highest_alpha(self, alpha):
        alpha                 = torch.squeeze(alpha)
        clusterOfHighestAlpha = torch.argmax(alpha)
        return self.extract_E_of_cluster_index(clusterOfHighestAlpha)
    
    def extract_nodeMean_of_highest_alpha(self, alpha):
        alpha                 = torch.squeeze(alpha)
        clusterOfHighestAlpha = torch.argmax(alpha)
        return self.extract_nodeMean_of_cluster_index(clusterOfHighestAlpha)
    
    def extract_D_E_nodeMean_of_highest_alpha(self, alpha):
        D = self.extract_D_of_highest_alpha(alpha)
        E = self.extract_E_of_highest_alpha(alpha)
        nodeMean = self.extract_nodeMean_of_highest_alpha(alpha)
        return D, E, nodeMean
    
    ###########################################################################
    # Extracting the D and E values for a selected cluster chosen through its
    # index 'clusterIndex'.
    
    def extract_D_of_cluster_index(self, clusterIndex):
        return torch.unsqueeze(self.D[int(clusterIndex), :], 0)
    
    def extract_E_of_cluster_index(self, clusterIndex):
        return torch.unsqueeze(self.E[int(clusterIndex), :], 0)
    
    def extract_nodeMean_of_cluster_index(self, clusterIndex):
        return torch.unsqueeze(self.nodesMean[int(clusterIndex), :], 0)
    
    def extract_D_E_nodeMean_of_cluster_index(self, clusterIndex):
        D = self.extract_D_of_cluster_index(clusterIndex)
        E = self.extract_E_of_cluster_index(clusterIndex)
        nodeMean = self.extract_nodeMean_of_cluster_index(clusterIndex)
        return D, E, nodeMean
    
    ###########################################################################
    
    def findParamsValueGivenMatricesDAndE(self,D,E,nodesMean,mu_t):
        
        # Prediction of parameters
        # Predicted parameters = matrix D * updated state z
        # So multiplying:
        # D: 
        #     (number of clusters x 
        #     batch size x 
        #     number of parameters x 
        #     z_dim)
        # By:
        # mu unsqueezed:
        #     (batch size x
        #      number of parameters)
        # Plus:
        # E:
        #    (number of clusters x
        #     batch size x
        #     number of parameters)
        
        params_pred = torch.squeeze(torch.matmul(D, torch.unsqueeze(mu_t, 2))) + E + nodesMean
        
        '''
        if mu_t.shape[0] == 1: # In case of batch_size = 0
            params_pred = torch.unsqueeze(torch.squeeze(torch.matmul(D, torch.unsqueeze(mu_t, 2))),1) + E + nodesMean
        else:
            params_pred = torch.squeeze(torch.matmul(D, torch.unsqueeze(mu_t, 2))) + E + nodesMean
        '''
        
        return params_pred
    
    def findParamsValue(self, alpha, mu_t):
            
        D = torch.matmul(alpha.float(), torch.reshape(self.D, [-1, self.clusteringDimension*self.dim_z]))
        D = torch.reshape(D, [-1, self.clusteringDimension, self.dim_z]) 
        D.view([mu_t.size()[0], self.clusteringDimension, self.dim_z])  
        D = D.float()     
        
        E = torch.matmul(alpha.float(), torch.reshape(self.E, [-1, self.clusteringDimension]))
        E = torch.reshape(E, [-1, self.clusteringDimension]) 
        E.view([mu_t.size()[0], self.clusteringDimension])  
        E = E.float()  
        
        nodesMean = torch.matmul(alpha.float(), torch.reshape(self.nodesMean, [-1, self.clusteringDimension]))
        nodesMean = torch.reshape(nodesMean, [-1, self.clusteringDimension]) 
        nodesMean.view([mu_t.size()[0], self.clusteringDimension])  
        nodesMean = nodesMean.float()  
        
        params_pred = self.findParamsValueGivenMatricesDAndE(D,E,nodesMean,mu_t)
        
        return params_pred
    
    ###########################################################################
    # FORWARD STEP OF KALMAN SMOOTHER
    
    def PrepareInputsForForwardLoop(self):
        
        # Concatenate the inputs
        inputs = self.y
        
        # Take the distance from the clusters for time instant 0 of the sequence.
        # Remember self.dist is done as follows:
        # (batch size, sequence length, number_of_clusters)
        _dist = self.dist[:, 0, :]

        # Dummy matrix to initialize B and C in scan
        dummy_init_A = torch.ones([self.Sigma.size()[0], self.dim_z, self.dim_z])
        dummy_init_B = torch.ones([self.Sigma.size()[0], self.dim_z])
        dummy_init_C = torch.ones([self.Sigma.size()[0], self.dim_y, self.dim_z])
        
        # Initialize to zero the time counter for the batch
        timeInBatchCounter = 0
        
        # Change the order of the inputs
        inputs = inputs.permute(1, 0, 2)
        
        # This is the value to be passed over the for loop
        # IF FIRST TIME INSTANT
        if self.firstTimeInstant == True:
            print('First element of the sequence')
            # Get probability of each discrete value
            alpha          = (self.alphaDistProb(_dist)).clone()
            self.updateAlphaBeginning(alpha)
            # Init values
            a_ = (self.mu, self.Sigma, self.mu, self.Sigma, alpha, 
                  dummy_init_A, dummy_init_B, dummy_init_C, timeInBatchCounter)   
        # IF NOT FIRST TIME INSTANT
        else:
            print('Not first element of the sequence')
            # Get probability of each discrete value
            alpha = self.alphaPrev.clone()
            # Init values
            a_ = (self.previousMu, self.previousSigma, self.previousMu, self.previousSigma, 
                  alpha, dummy_init_A, dummy_init_B, dummy_init_C, timeInBatchCounter)
        
        return self.y, inputs, a_

    # Function to perform the forward part of the Kalman Filter, i.e., the 
    # filtering part only.
    # OUTPUTS:
    # - forward_states_all: this is a set of features related to filtering.
    #   More precisely, they contain:
    #   mu_pred: prediction mean on KF
    #          It has dimension [sequence_length, batch_size, z_dim]
    #   Sigma_pred: prediction covariance on KF 
    #          It has dimension [sequence_length, batch_size, z_dim, z_dim]
    #   mu_t: filtered mean on KF 
    #   Sigma_t: filtered covarianc on KF
    #   alpha: distance value from clusters
    #   u: control (not really an output)
    #   A: transition matrices
    #   B: control matrices
    #   C: pseudo-observation matrices
    def compute_forwards(self, reuse=None):
        
        # Note:
        # self.mu    -> a_[0].shape -> torch.Size([sequence len, z dim])
        # self.sigma -> a_[1].shape -> torch.Size([sequence len, z dim, z dim])
        y, inputs, a_ = self.PrepareInputsForForwardLoop()
        
        # Looping over the sequence
        for i in range(0, len(inputs)):
            
            # CALL FORWARD FUNCTION
            # Give as input 'a' and 'inputs'
            forward_states      = self.forward_step_fn(a_, inputs[i]) # <------------------
            base_forward_states = forward_states[0:-1]
            a_                  = base_forward_states
            
            # CREATE ARRAY
            # Add one dimension at the beginning, to account for sequence length
            base_forward_states_redimensioned, params_pred_redimensioned = \
               self.AddDimensionAtBeginningOfForwardFeatures(forward_states)
            
            if i == 0:
                # Initialize a set of features each constituted by a vector
                # where the first dimension is related to the sequence length
                base_forward_states_all, params_pred_all = \
                   self.InitializeForwardFeatures(base_forward_states_redimensioned,params_pred_redimensioned)  
            else:
                
                # Concatenate current features in the vector
                base_forward_states_all, params_pred_all = \
                   self.ConcatenateForwardFeatures(base_forward_states_all, base_forward_states_redimensioned,
                                                   params_pred_all,params_pred_redimensioned)
                
        # Update the value of previous mu and sigma to use for next batch
        self.updatePreviousMuAndPreviousSigma(base_forward_states_all)
        self.updatePreviousY(y[:, -1, :])
        self.setNotFirstTimeInstantOfSequence()
        
        return (base_forward_states_all, params_pred_all)
    
    def forward_step_fn(self, params , inputs):
        
        output_base_function = super(KalmanFilter_D, self).forward_step_fn(params , inputs) 
        mu_pred, Sigma_pred, mu_t, Sigma_t, alpha, A, B, C, timeInBatchCounter = output_base_function
        
        # -+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # Now through matrix D obtain the parameters values from the update
        params_pred = self.findParamsValue(alpha, mu_t)
        
        return mu_pred, Sigma_pred, mu_t, Sigma_t, alpha, A, B, C, timeInBatchCounter, params_pred
    
    ###########################################################################
    # BACKWARD STEP OF KALMAN SMOOTHER
        
    def compute_backwards(self, forward_states_all):
        
        base_forward_states_all, params_preds = forward_states_all
        
        # Call to base function
        backward_states_all_reordered, As, Bs, Cs, alphas  = \
           super(KalmanFilter_D, self).compute_backwards(base_forward_states_all) 
           
        return backward_states_all_reordered, As, Bs, Cs, alphas, params_preds
    
    ###########################################################################
    # Adding dimensions while looping and initializing
    
    def AddDimensionAtBeginningOfForwardFeatures(self, forward_states):
        
        # All tuples except the last one (this can be given to the base function)
        base_forward_states = forward_states[0:-1]
        
        # Last tuple only (predicted parameters)
        params_pred         = forward_states[len(forward_states)-1]
        
        # Call to base function
        base_forward_states = \
           super(KalmanFilter_D, self).AddDimensionAtBeginningOfForwardFeatures(base_forward_states) 
        
        # For last features  
        params_pred = torch.unsqueeze(params_pred, 0)

        return base_forward_states, params_pred
    
    def InitializeForwardFeatures(self, base_forward_states, params_pred):
       
        # Call to base function
        base_forward_states = super(KalmanFilter_D, self).InitializeForwardFeatures(base_forward_states) 
        # For last features        
        params_preds = params_pred
        
        return base_forward_states, params_preds
    
    def ConcatenateForwardFeatures(self, base_forward_states_all, base_forward_states, params_preds, params_pred):
        
        # Call to base function
        base_forward_states_all     = super(KalmanFilter_D, self).ConcatenateForwardFeatures(
                base_forward_states_all, base_forward_states) 
        # For last features 
        params_preds    = torch.cat([params_preds, params_pred], axis=0)
        
        return base_forward_states_all, params_preds
    
    # Function to update mu and sigma 
    def updatePreviousMuAndPreviousSigma(self, base_forward_states):

        # Call base
        super(KalmanFilter_D, self).updatePreviousMuAndPreviousSigma(base_forward_states) 
        
        return
    
    ###########################################################################
    # Filtering and smoothing (this are the main functions from which all starts)
    
    # Function to perform FILTERING (so forward part only)
    def filter(self):
        mu_pred, Sigma_pred, mu_filt, Sigma_filt, alpha, A, B, C, params_pred = forward_states = \
            self.compute_forwards(reuse=True)
            
        # Swap batch dimension and time dimension
        mu_filt    = mu_filt.permute(1, 0, 2)
        Sigma_filt = Sigma_filt.permute(1, 0, 2, 3)
        forward_states = (mu_filt, Sigma_filt)
        
        return mu_pred, Sigma_pred, tuple(forward_states), A.permute(1, 0, 2, 3), B.permute(1, 0, 2), \
               C.permute(1, 0, 2, 3), alpha.permute(1, 0, 2), params_pred.permute(1, 0, 2)
        
    # Function to perform SMOOTHING (so forward part + backward part)
    def smooth(self):
        # first perform the predictions from the current time
        # and then compute backwards
        # forward + backwards = KALMAN SMOOTHING
        # backwards_states    = smoothed mu + sigma
        
        base_forward_states_all, params_pred_all = self.compute_forwards()
        mu_preds, Sigma_preds, mu_filts, Sigma_filts, alphas, As, Bs, Cs = base_forward_states_all
        
        backward_states, A, B, C, alpha, params_preds = self.compute_backwards((base_forward_states_all, params_pred_all))  
        mus, sigmas = backward_states
        
        # Permute dimensions to have the batch size as first dimension and sequence
        # length as second dimension
        mus    = mus.permute(1, 0, 2, 3)#, [1, 0, 2])
        sigmas = sigmas.permute(1, 0, 2, 3)#, [1, 0, 2, 3])
        
        backward_states = (mus, sigmas)
        
        # Define return values
        return_values = tuple(backward_states), A.permute(1, 0, 2, 3), B.permute(1, 0, 2), \
                   C.permute(1, 0, 2, 3), alpha.permute(1, 0, 2), params_preds.permute(1, 0, 2), \
                   mu_preds, Sigma_preds, mu_filts, Sigma_filts
               
        return return_values