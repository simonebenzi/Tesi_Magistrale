# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:39:07 2021

@author: Abrham
"""

import torch
import torch.nn as nn

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

class KalmanFilter(nn.Module):
    
    # Initialization of the filter.
    # INPUTS:
    # - dim_z: dimension of state 'z' (smaller)
    # - dim_y: dimension of state 'a' (bigger)
    # - num_clusters: number of clusters of odometry
    # - sequence_length: length of the sequences
    # - init_kf_matrices
    # - init_cov: initial covariance value
    # - batch_size: batch size
    # - noise_transition: noise for the transition covariance
    # - noise_emission: noise for the emission covariance
    def __init__(self, dim_z, dim_y, num_clusters, sequence_length, init_kf_matrices, \
                 init_cov, batch_size, noise_transition, noise_emission):
        super(KalmanFilter, self).__init__()
        
        print("Initialization of Kalman Filter")
        
        self.dim_z = dim_z
        self.dim_y = dim_y # = dim_a
        self.num_clusters     = num_clusters
        self.init_kf_matrices = init_kf_matrices
        self.init_cov         = init_cov
        self.batch_size       = batch_size
        self.noise_transition = noise_transition
        self.noise_emission   = noise_emission
        self.sequence_length  = sequence_length
        
        self.consider_A = True
        self.consider_B = True
        
        self.multiple_Cs = True
        
        # Initializers for LGSSM variables. A is intialized with identity 
        # matrices, B and C randomly from a gaussian
        # Transition matrix
        self.A = torch.eye(self.dim_z) 
        self.A = self.A.unsqueeze(0).repeat(self.num_clusters, 1, 1).to(device)
        if self.consider_A == True:
            self.A = nn.Parameter(self.A, requires_grad=True)
        else:
            self.A = nn.Parameter(self.A, requires_grad=False)
        # Control matrix
        if self.consider_B == True:
            self.B = torch.zeros(self.num_clusters, self.dim_z).to(device)
            #self.B = init_kf_matrices*torch.randn(self.num_clusters, self.dim_z).to(device)
            self.B = nn.Parameter(self.B, requires_grad=True)
        else:
            self.B = torch.zeros(self.num_clusters, self.dim_z).to(device)
            self.B = nn.Parameter(self.B, requires_grad=False)
        # Pseudo-observation matrix
        if self.multiple_Cs == True:
            self.C = nn.Parameter(init_kf_matrices*torch.randn(self.num_clusters, self.dim_y, self.dim_z).to(device), requires_grad=True)
        else:
            self.C = nn.Parameter(init_kf_matrices*torch.randn(self.dim_y, self.dim_z).to(device), requires_grad=True)
            
        # We use isotropic covariance matrices
        self.Q = self.noise_transition * torch.eye(self.dim_z).to(device)
        self.R = self.noise_emission * torch.eye(self.dim_y).to(device)

        # p(z_1)
        # Mean
        self.mu    = torch.zeros((self.batch_size, self.dim_z)).to(device)
        # Covariance
        self.Sigma = self.init_cov*torch.eye(self.dim_z)
        self.Sigma = self.Sigma.unsqueeze(0).repeat(self.batch_size, 1, 1).to(device)           

        # identity matrix
        self._I   = torch.eye(dim_z).to(device)
        
        # These will need to be added before a call to forward
        self.dist = torch.zeros(self.batch_size,self.sequence_length,self.num_clusters).to(device)
        self.y    = torch.zeros(self.batch_size,self.sequence_length,self.dim_y).to(device)
        
        # This is a flag to tell whether this is the first time instant of a 
        # sequence or not
        self.firstTimeInstant = True
        
        return
    
    # Function to change the batch size (in case testing has different batch
    # size from training).
    def ChangeBatchSize(self, new_batch_size):
        
        self.batch_size = new_batch_size
        
        self.mu    = torch.zeros((self.batch_size, self.dim_z)).to(device)
        # Covariance
        self.Sigma = self.init_cov*torch.eye(self.dim_z)
        self.Sigma = self.Sigma.unsqueeze(0).repeat(self.batch_size, 1, 1).to(device)
        
        self.dist = torch.zeros(self.batch_size,self.sequence_length,self.num_clusters).to(device)
        self.y    = torch.zeros(self.batch_size,self.sequence_length,self.dim_y).to(device)
        
        return
        
    # This function inizializes the variables related to a single call of the Kalman
    # Filter, i.e.:
    # - the distances from clusters dist
    # - the a/y states encoded with the VAE for the current batch
    # - the odometries of the current batch
    def initializeCallToKalmanFilter(self, dist, y, alphaDistProb):
        
        # distances from clusters
        self.dist  = dist.detach()
        # states a/y
        self.y     = y.detach()
        # alpha function
        self.alphaDistProb = alphaDistProb
        
        return
    
    # Function to set that this is the first time instant of the sequence
    def setFirstTimeInstantOfSequence(self):
        # This is a flag to tell whether this is the first time instant of a 
        # sequence or not
        self.firstTimeInstant = True       
        return
    
    # Function to set that this is not the first time instant of the sequence
    def setNotFirstTimeInstantOfSequence(self):
        # This is a flag to tell whether this is the first time instant of a 
        # sequence or not
        self.firstTimeInstant = False       
        return
        
    # Function to update mu and sigma 
    def updatePreviousMuAndPreviousSigma(self, forward_states):
        
        # RETRIEVE THE VALUES FROM FORWARD
        # Retrieve features
        mu_preds, Sigma_preds, mu_filts, Sigma_filts, alphas, As, Bs, Cs = forward_states
        
        # We need to retrieve the very last element of mu_filt and sigma_filt to
        # use as initializer for the backward pass
        mu_filt_lastTimeInstant    = mu_filts[-1, :, :]
        Sigma_filt_lastTimeInstant = Sigma_filts[-1, :, :]
        
        # Update
        self.previousMu    = mu_filt_lastTimeInstant.detach()
        self.previousSigma = Sigma_filt_lastTimeInstant.detach()
        
        return
    
    def updateAlphaPrev(self, alphaPrev):
        self.alphaPrev = alphaPrev.detach()
        return
    
    def updateAlphaBeginning(self, alphaBeginning):
        self.alphaBeginning = alphaBeginning.detach()
        return
    
        # Function to update mu and sigma 
    def updatePreviousY(self, ys_lastTimeInstant):
        self.previousY  = ys_lastTimeInstant.detach()        
        return
    
   ###########################################################################
    # Extracting the A.B and C values for the cluster where alpha is maximum.
    
    def extract_C_of_highest_alpha(self, alpha):        
        alpha                 = torch.squeeze(alpha)
        clusterOfHighestAlpha = torch.argmax(alpha) 
        return self.extract_C_of_cluster_index(clusterOfHighestAlpha)
    
    def extract_B_of_highest_alpha(self, alpha):        
        alpha                 = torch.squeeze(alpha)
        clusterOfHighestAlpha = torch.argmax(alpha)        
        return self.extract_B_of_cluster_index(clusterOfHighestAlpha)
    
    def extract_A_of_highest_alpha(self, alpha):        
        alpha                 = torch.squeeze(alpha)
        clusterOfHighestAlpha = torch.argmax(alpha)        
        return self.extract_A_of_cluster_index(clusterOfHighestAlpha)
    
    def extract_A_B_C_of_highest_alpha(self, alpha):        
        A = self.extract_A_of_highest_alpha(alpha)
        B = self.extract_B_of_highest_alpha(alpha)
        C = self.extract_C_of_highest_alpha(alpha)       
        return A, B, C
    
    ###########################################################################
    # Extracting the A, B and C values for a selected cluster chosen through its
    # index 'clusterIndex'.
    
    def extract_C_of_cluster_index(self, clusterIndex):   
        if self.multiple_Cs == True:
            return torch.unsqueeze(self.C[int(clusterIndex), :], 0)
        else:
            return torch.unsqueeze(self.C[:], 0)
            
    
    def extract_B_of_cluster_index(self, clusterIndex):               
        return torch.unsqueeze(self.B[int(clusterIndex), :], 0)
    
    def extract_A_of_cluster_index(self, clusterIndex):              
        return torch.unsqueeze(self.A[int(clusterIndex), :], 0)
    
    def extract_A_B_C_of_cluster_index(self, clusterIndex):        
        A = self.extract_A_of_cluster_index(clusterIndex)
        B = self.extract_B_of_cluster_index(clusterIndex)
        C = self.extract_C_of_cluster_index(clusterIndex)       
        return A, B, C
    
    ###########################################################################
    # PSEUDO-EMISSION MODEL
    
    # Function to find the state 'a' from state 'z' and a given matrix C.
    @staticmethod
    def ProjectZStateToAStateGivenCMatrix(z_state, C):
        
        a_state = torch.squeeze(torch.matmul(C, torch.unsqueeze(z_state, 2)))  # (bs, dim_y)        
        return a_state
    
    # Function to find the state 'a' from state 'z', taking the C value of a cluster.
    def ObtainAValueFromZValueGivenCluster(self, z_state, cluster):
        
        # A, B, C, D, E, and nodeMean matrix for the current cluster
        A,B,C = self.extract_A_B_C_of_cluster_index(cluster)        
        # Prediction
        a_state = KalmanFilter.ProjectZStateToAStateGivenCMatrix(z_state, C)        
        return a_state
    
    # Perform update phase given the matrix C.
    # INPUTS:
    # - Sigma_pred: predicted covariance,
    # - mu_pred: predicted mean,
    # - y: observation
    # - C: given matrix C
    def perform_update_given_C(self, Sigma_pred, mu_pred, y, C):
        
        if mu_pred.shape[0] != self.batch_size and self.batch_size == 1:
            mu_pred = torch.unsqueeze(mu_pred, 0)
        
        # Residual
        # Get y from mu and C
        # a_t/t-1 = C * z_t/t-1
        y_pred = KalmanFilter.ProjectZStateToAStateGivenCMatrix(mu_pred, C)
        # Calculate the innovation
        # inn_t     = a_t - a_t/t-1
        innovation = y - y_pred  # (bs, dim_y)
        
        # Project system uncertainty into measurement space, i.e., innovation covariance
        # S_t     = C * P_t/t-1 * C^T + R
        S     = torch.matmul(torch.matmul(C, Sigma_pred),C.permute(0, 2, 1)) + self.R  # (bs, dim_y, dim_y)
        # Inverse of the innovation covariance
        # S^-1
        S_inv = torch.inverse(S)
        
        # Calculate the Kalman Gain
        # K_t     = P_t/t-1* C^T * S^-1
        K = torch.matmul(torch.matmul(Sigma_pred, C.permute(0, 2, 1)), S_inv)  # (bs, dim_z, dim_y)

        # Updated mean of z
        # z_t/t   = z_t/t-1 + K_t * y_t
        mu_t    = mu_pred + torch.squeeze(torch.matmul(K, torch.unsqueeze(innovation, 2)))  # (bs, dim_z)
        # Updated covariance of z
        # P_t/t   = ( I - K_t * C ) * P_t/t-1
        I_KC    = self._I - torch.matmul(K, C)  # (bs, dim_z, dim_z)
        Sigma_t = torch.matmul(I_KC, Sigma_pred)
        #Sigma_t = torch.matmul(torch.matmul(I_KC, Sigma_pred), I_KC.permute(0, 2, 1)) + self._sast(self.R, K) # (bs, dim_z, dim_z)
        
        return mu_t, Sigma_t
    
    # Perform update phase calculating the matrix C from the 
    # INPUTS:
    # - Sigma_pred: predicted covariance,
    # - mu_pred: predicted mean,
    # - y: observation
    # - C: given matrix C
    def perform_update(self, alpha, Sigma_pred, mu_pred, y):
        
        # Mixture of C:
        # Multiply C with alpha. In order for this to be possible, C must
        # be modified with reshape in order to repeat it K (= number of clusters) 
        # times (-> -1 becomes = K)
        if self.multiple_Cs == True:
            C = torch.matmul(alpha.double(), torch.reshape(self.C.double(), (-1, self.dim_y*self.dim_z)))  #(bs, k) x (k, dim_y*dim_z) -> (bs, dim_y*dim_z)
            # Reshape
            C = torch.reshape(C, [-1, self.dim_y, self.dim_z])  # (bs, dim_y, dim_z)
            C.view([Sigma_pred.size()[0], self.dim_y, self.dim_z])
        else:
            C = self.C.double()
            C = torch.unsqueeze(C,0)
            C = C.repeat(Sigma_pred.size()[0], 1, 1)
        
        C = C.float()
    
        mu_t, Sigma_t = self.perform_update_given_C(Sigma_pred, mu_pred, y, C)
        
        return mu_t, Sigma_t, C
        
    def update_alpha_CG_KVAE(self, input_for_alpha_update):
        
        _dist, alpha     = input_for_alpha_update
        # RETRIEVE ALPHA FROM CLUSTERS DISTANCES
        alpha            = self.alphaDistProb(_dist, alpha)
        self.updateAlphaPrev(alpha)
        return alpha
    
    ###########################################################################
    # TRANSITION MODEL
    
    def perform_prediction_given_A_B(self, mu_t, Sigma_t, A, B):
        
        mu_pred    = torch.squeeze(torch.matmul(A, torch.unsqueeze(mu_t, 2))) + torch.squeeze(B)
        
        '''
         # Perform prediction of mean
        if self.consider_A == True and self.consider_B == True:
            mu_pred    = torch.squeeze(torch.matmul(A, torch.unsqueeze(mu_t, 2))) + torch.squeeze(B)
        elif self.consider_A == True:
            mu_pred    = torch.squeeze(torch.matmul(A, torch.unsqueeze(mu_t, 2)))
        elif self.consider_B == True:
            mu_pred    = mu_t + torch.squeeze(B)
            
        if mu_pred.shape[0] != self.batch_size and self.batch_size == 1:
            mu_pred = torch.unsqueeze(mu_pred, 0)
        '''
        
        # Perform prediction of covariance
        Sigma_pred = torch.matmul(torch.matmul(A, Sigma_t), A.permute(0, 2, 1)) + self.Q
        
        return mu_pred, Sigma_pred
        
    def perform_prediction(self, alpha, mu_t, Sigma_t):
        
        # Mixture of A
        A = torch.matmul(alpha.float(), torch.reshape(self.A, [-1, self.dim_z*self.dim_z]))  # (bs, k) x (k, dim_z*dim_z)
        A = torch.reshape(A, [-1, self.dim_z, self.dim_z])  # (bs, dim_z, dim_z)
        A.view(Sigma_t.size())  # set shape to batch_size x dim_z x dim_z

        # Mixture of B
        B = torch.matmul(alpha.float(), torch.reshape(self.B, [-1, self.dim_z]))  # (bs, k) x (k, dim_y*dim_z)
        B = torch.reshape(B, [-1, self.dim_z])  # (bs, dim_y, dim_z)
        B.view([A.size()[0], self.dim_z])
        
        A = A.float()
        B = B.float()
        
        mu_pred, Sigma_pred = self.perform_prediction_given_A_B(mu_t, Sigma_t, A, B)
            
        return mu_pred, Sigma_pred, A, B
    
    ###########################################################################
    # FORWARD STEP OF KALMAN SMOOTHER
    
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
        
        # Concatenate the inputs
        inputs = self.y
        # Change the order of the inputs
        inputs = inputs.permute(1, 0)
        
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
        
        # This is the value to be passed over the for loop
        # IF FIRST TIME INSTANT
        if self.firstTimeInstant == True:
            print('First element of the sequence')
            # Get probability of each discrete value
            alpha          = (self.alphaDistProb(_dist))
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
            
        print('Number of time instants in batch: ' + str(len(inputs)))
        
        for i in range(0, len(inputs)):
            
            # CALL FORWARD FUNCTION
            # Give as input 'a' and 'inputs'
            forward_states = self.forward_step_fn(a_, inputs[i]) # <------------------
            a_             = forward_states
            
            # CREATE ARRAY
            # Add one dimension at the beginning, to account for sequence length
            forward_states_redimensioned = self.AddDimensionAtBeginningOfForwardFeatures(forward_states)
            if i == 0:
                # Initialize a set of features each constituted by a vector
                # where the first dimension is related to the sequence length
                forward_states_all = self.InitializeForwardFeatures(forward_states_redimensioned)  
            else:
                # Concatenate current features in the vector
                forward_states_all = self.ConcatenateForwardFeatures(forward_states_all, forward_states_redimensioned)
                
        # Update the value of previous mu and sigma to use for next batch
        self.updatePreviousMuAndPreviousSigma(forward_states_all)
        self.updatePreviousY(self.y[:, -1, :])
        self.setNotFirstTimeInstantOfSequence()
        
        return forward_states_all

    def forward_step_fn(self, params , inputs):
        
        # RETRIEVING INPUTS
        
        # Extract the values of last time instant from the "params" vector
        mu_pred, Sigma_pred, _, _, alpha, _, _, _, timeInBatchCounter = params
        
        y = self.ExtractInputsOfForwardStep(inputs)
        
        # Retrieve the distance from the clusters
        # Dimension:
        # (batch size, number of clusters)
        _dist = self.dist[:,timeInBatchCounter,:]
        
        # -+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # UPDATE PHASE
        mu_t, Sigma_t, C          = self.perform_update(alpha, Sigma_pred, mu_pred, y)
        # Alpha
        input_for_alpha_update = _dist, alpha
        alpha                  = self.update_alpha_CG_KVAE(input_for_alpha_update)
        
        # -+-+-+-+-+-+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        # PREDICTION PHASE
        mu_pred, Sigma_pred, A, B = self.perform_prediction(alpha, mu_t, Sigma_t)
        
        # Increment counter related to time in the batches sequence
        timeInBatchCounter +=1 
        
        return mu_pred, Sigma_pred, mu_t, Sigma_t, alpha, A, B, C, timeInBatchCounter
    
    def ExtractInputsOfForwardStep(self, inputs):
        
        # Select the first dim_y elements of the inputs vector, over the full batch
        # Dimension:
        # (batch size, dim_y)
        y     = inputs[: , 0:self.dim_y]
        
        return y
            
    ###########################################################################
    # BACKWARD STEP OF KALMAN SMOOTHER
    
    # Function to perform the backward part of the Kalman Filter, i.e., the 
    # smoothing part.
    # OUTPUTS:
    # - backward_states_all_reordered: this is a set of features related to 
    #   performing the smoothing.
    #   mu_backs: smoothed mean on KF
    #          It has dimension [sequence_length, batch_size, z_dim]
    #   Sigma_backs: smoothed covariance on KF 
    #          It has dimension [sequence_length, batch_size, z_dim, z_dim]
    # - As, from filtering
    # - Bs, from filtering 
    # - Cs, from filtering
    # - alphas, from filtering
    def compute_backwards(self, forward_states):
        
        '''
        What this function does step by step:
        - Reads forward_states, which is a set of arrays
        - Retrieves the last filtered values of the sequence
        - Does backward pass givinf the filtered/predicted values at each
          time instant, starting from the second to last. The last filtered values
          are given too in the first call of the for loop. In the following calls
          the output of the previous backward step is given instead.
        - The found array of smoothed values is set in the forward order again.
        - The last filtered values from forward pass are added at the end.
        INPUT:
            - list of filtered values and other features from forward pass.
              Length of the list = sequence length.
        OUTPUT:
            - list of smoothed values and other features from backward pass.
              Length of the list = sequence length.            
        '''
        
        # RETRIEVE THE VALUES FROM FORWARD
        mu_preds, Sigma_preds, mu_filts, Sigma_filts, alphas, As, Bs, Cs = forward_states

        # Add one dimension to the two features of the state (mu_pred and mu_filt)
        # This will be necessary in the 'backward_step_fn' function because
        # there will be a multiplication between tensors of dimensions e.g.
        # [batch_size, z_dim, z_dim]  <- from sigma
        # and 
        # [batch_size, z_dim, 1] <- from mu. So that 1 is necessary
        mu_preds = torch.unsqueeze(mu_preds, 3)
        mu_filts = torch.unsqueeze(mu_filts, 3)
        
        # We need to retrieve the very last element of mu_filt and sigma_filt to
        # use as initializer for the backward pass
        mu_filt_lastTimeInstant    = mu_filts[-1, :, :, :]
        Sigma_filt_lastTimeInstant = Sigma_filts[-1, :, :, :]
        initializer                = (mu_filt_lastTimeInstant, Sigma_filt_lastTimeInstant)

        # Now we need to perform a loop from the second to last,
        # to the first elements of the sequence
        sequenceLength = mu_filts.shape[0]
        # For initializing backward pass we use the filtered values at last 
        # time step of forward pass
        smoothedValuesFromPreviousStep = initializer
        # E.g., if sequence length = 20:
        # range(18, -1, -1)
        # 18 (second to last), 17, 16, 15 ... 0
        for i in range(sequenceLength-2 , -1, -1):
            # RETRIEVE FEATURES
            # Select necessary features from forward at time instant i
            mu_pred     = mu_preds[i, :, :, :]
            Sigma_pred  = Sigma_preds[i, :, :, :]
            mu_filt     = mu_filts[i, :, :, :]
            Sigma_filt  = Sigma_filts[i, :, :, :]
            A           = As[i, :, :, :]
            # To give as input to backward_step_fn
            states_scan = (mu_pred, Sigma_pred, mu_filt, Sigma_filt, A)
            
            # CALL BACKWARD FUNCTION
            backward_states = self.backward_step_fn(smoothedValuesFromPreviousStep, states_scan) # <------------------
            # To give as input at next call of backward_step_fn
            smoothedValuesFromPreviousStep = backward_states
            
            # CREATE ARRAY
            # Add one dimension at the beginning, to account for sequence length
            backward_states_redimensioned  = KalmanFilter.AddDimensionAtBeginningOfBackwardFeatures(backward_states)
            if i == sequenceLength-2:
                # Initialize a set of features each constituted by a vector
                # where the first dimension is related to the sequence length
                backward_states_all = KalmanFilter.InitializeBackwardFeatures(backward_states_redimensioned)  
            else:
                # Concatenate current features in the vector
                backward_states_all = KalmanFilter.ConcatenateBackwardFeatures(backward_states_all, backward_states_redimensioned)
            
        # Take out smoothed mu and sigma
        mu_backs, Sigma_backs = backward_states_all
        
        # Flip order along sequence dimension, to go from first to last again
        mu_backs    = torch.flip(mu_backs, [0])
        Sigma_backs = torch.flip(Sigma_backs, [0])
        
        # Add the final state from the filtering distribution
        mu_backs    = torch.cat([mu_backs,    torch.unsqueeze(mu_filt_lastTimeInstant, 0)],    axis=0)
        Sigma_backs = torch.cat([Sigma_backs, torch.unsqueeze(Sigma_filt_lastTimeInstant, 0)], axis=0)
        
        backward_states_all_reordered = (mu_backs, Sigma_backs)
        
        return backward_states_all_reordered, As, Bs, Cs, alphas
    
    def backward_step_fn(self, params, inputs):

        # RETRIEVING
        # The output from the previous backward pass
        # (which , if this is the first time instant of backward, corresponds
        # to the last output of forward)
        mu_back, Sigma_back = params
        # The predictions and filtering at current time instant in forward pass
        mu_pred_tp1, Sigma_pred_tp1, mu_filt_t, Sigma_filt_t, A = inputs
        
        # PERFORMING SMOOTHING
        J_t = torch.matmul(A.permute(0,2,1), torch.inverse(Sigma_pred_tp1))
        J_t = torch.matmul(Sigma_filt_t, J_t)

        mu_back    = mu_filt_t + torch.matmul(J_t, mu_back - mu_pred_tp1)
        conj_value = torch.conj(J_t.permute(0,2,1))#.cpu()).to(device) # This is to be able to use conj function also on GPU case for pytorch 1.4.1
        Sigma_back = Sigma_filt_t + torch.matmul(J_t, torch.matmul(Sigma_back - Sigma_pred_tp1, conj_value)) 

        return mu_back, Sigma_back
    
    ###########################################################################
    # Adding dimensions while looping and initializing
    
    def AddDimensionAtBeginningOfForwardFeatures(self, forward_states):
        
        # Retrieve features
        mu_pred, Sigma_pred, mu_t, Sigma_t, alpha, A, B, C, _ = forward_states
        
        # Unsqueeze them to 
        mu_pred        = torch.unsqueeze(mu_pred, 0)
        Sigma_pred     = torch.unsqueeze(Sigma_pred, 0)
        mu_t           = torch.unsqueeze(mu_t, 0)
        Sigma_t        = torch.unsqueeze(Sigma_t, 0)
        alpha          = torch.unsqueeze(alpha, 0)
        A              = torch.unsqueeze(A, 0)
        B              = torch.unsqueeze(B, 0)
        C              = torch.unsqueeze(C, 0)
        
        return mu_pred, Sigma_pred, mu_t, Sigma_t, alpha, A, B, C
    
    @staticmethod
    def AddDimensionAtBeginningOfBackwardFeatures(backward_states):
        
        # Retrieve features
        mu_back, Sigma_back = backward_states
        
        # Unsqueeze them to 
        mu_back    = torch.unsqueeze(mu_back, 0)
        Sigma_back = torch.unsqueeze(Sigma_back, 0)

        return mu_back, Sigma_back
    
    def InitializeForwardFeatures(self, forward_states):
        
        # Retrieve features
        mu_pred, Sigma_pred, mu_t, Sigma_t, alpha, A, B, C = forward_states
        
        # Use the retrieved features for initialization
        mu_preds    = mu_pred
        Sigma_preds = Sigma_pred
        mu_ts       = mu_t
        Sigma_ts    = Sigma_t
        alphas      = alpha
        As          = A
        Bs          = B
        Cs          = C
        
        return mu_preds, Sigma_preds, mu_ts, Sigma_ts, alphas, As, Bs, Cs
    
    @staticmethod
    def InitializeBackwardFeatures(backward_states):
        
        # Retrieve features
        mu_back, Sigma_back = backward_states
        
        # Use the retrieved features for initialization
        mu_backs    = mu_back
        Sigma_backs = Sigma_back 
        
        return mu_backs, Sigma_backs  
    
    def ConcatenateForwardFeatures(self, forward_states_all, forward_states):
        
        # Retrieve features
        mu_preds, Sigma_preds, mu_ts, Sigma_ts, alphas, As, Bs, Cs = forward_states_all
        mu_pred, Sigma_pred, mu_t, Sigma_t, alpha, A, B, C = forward_states
        
        mu_preds        = torch.cat([mu_preds,    mu_pred],    axis=0)
        Sigma_preds     = torch.cat([Sigma_preds, Sigma_pred], axis=0)
        mu_ts           = torch.cat([mu_ts,       mu_t],       axis=0)
        Sigma_ts        = torch.cat([Sigma_ts,    Sigma_t],    axis=0)
        alphas          = torch.cat([alphas,      alpha],      axis=0)
        As              = torch.cat([As,          A],          axis=0)
        Bs              = torch.cat([Bs,          B],          axis=0)
        Cs              = torch.cat([Cs,          C],          axis=0)
        
        return mu_preds, Sigma_preds, mu_ts, Sigma_ts, alphas, As, Bs, Cs
    
    @staticmethod
    def ConcatenateBackwardFeatures(backward_states_all, backward_states):
        
        # Retrieve features
        mu_backs, Sigma_backs = backward_states_all
        mu_back, Sigma_back   = backward_states
        
        mu_backs    = torch.cat([mu_backs,    mu_back],    axis=0)
        Sigma_backs = torch.cat([Sigma_backs, Sigma_back], axis=0)
        
        return mu_backs, Sigma_backs
          
    ###########################################################################
    # Filtering and smoothing (this are the main functions from which all starts)
    
    # Function to perform FILTERING (so forward part only)
    def filter(self):
        mu_pred, Sigma_pred, mu_filt, Sigma_filt, alpha, A, B, C = forward_states = \
            self.compute_forwards(reuse=True)
            
        # Swap batch dimension and time dimension
        mu_filt    = mu_filt.permute(1, 0, 2)
        Sigma_filt = Sigma_filt.permute(1, 0, 2, 3)

        forward_states = (mu_filt, Sigma_filt)
        
        return mu_pred, Sigma_pred, tuple(forward_states), A.permute(1, 0, 2, 3), B.permute(1, 0, 2), \
               C.permute(1, 0, 2, 3), alpha.permute(1, 0, 2)
    
    # Function to perform SMOOTHING (so forward part + backward part)
    def smooth(self):
        # first perform the predictions from the current time
        # and then compute backwards
        # forward + backwards = KALMAN SMOOTHING
        # backwards_states    = smoothed mu + sigma
        
        forward_states = self.compute_forwards()
        mu_preds, Sigma_preds, mu_filts, Sigma_filts, alphas, As, Bs, Cs = forward_states
        
        backward_states, A, B, C, alpha = self.compute_backwards(forward_states)  
        mus, sigmas = backward_states
        
        # Permute dimensions to have the batch size as first dimension and sequence
        # length as second dimension
        mus    = mus.permute(1, 0, 2, 3)#, [1, 0, 2])
        sigmas = sigmas.permute(1, 0, 2, 3)#, [1, 0, 2, 3])
        
        backward_states = (mus, sigmas)
        
        # Define return values
        return_values = tuple(backward_states), A.permute(1, 0, 2, 3), B.permute(1, 0, 2), \
               C.permute(1, 0, 2, 3), alpha.permute(1, 0, 2), mu_preds, Sigma_preds, mu_filts, Sigma_filts
        return return_values