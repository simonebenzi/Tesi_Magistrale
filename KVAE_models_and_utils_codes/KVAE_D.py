
# This is a KVAE that additionally learns a matrix predicting from z to the 
# parameters state.
import torch
import torch.nn.functional as F

from KVAE_models_and_utils_codes import KVAE
from KVAE_models_and_utils_codes import p_filter_D

from ConfigurationFiles import Config_GPU as ConfigGPU

###############################################################################
# ---------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)

###############################################################################
# ------------------  KalmanVariationalAutoencoder   --------------------------
###############################################################################

class KalmanVariationalAutoencoder_D(KVAE.KalmanVariationalAutoencoder):
    """ This class defines functions to build, train and evaluate Kalman Variational Autoencoders
    """
    
    # Function to initialize the Kalman Filter subcomponent of the KVAE.
    # The difference with the base one is that we use a 'p_filter_D' object
    # instead of 'p_filter'.
    # INPUTS:
    # - config: configuration file
    # - clusterGraph: clustering object of odometry
    # - trainingData: data of training.
    # - sequence_length: length of sequences
    def BuildKFSubcomponent(self, config, clusterGraph, trainingData, sequence_length = None):
        
        if sequence_length == None:
            sequence_length = config['sequence_length']

        self.kf = p_filter_D.KalmanFilter_D(dim_z               = config['dim_z'],
                                            dim_y               = config['dim_a'],
                                            num_clusters        = clusterGraph.num_clusters,
                                            init_kf_matrices    = config['init_kf_matrices'],
                                            init_cov            = config['init_cov'],
                                            batch_size          = config['batch_size'],
                                            noise_transition    = config['noise_transition'],
                                            noise_emission      = config['noise_emission'], 
                                            sequence_length     = sequence_length,
                                            clusteringDimension = self.clusteringDimension,
                                            nodesMean           = torch.from_numpy(clusterGraph.nodesMean).float()).to(device)       
        return
    
    # Initialization
    # Inputs:
    # - config: configuration parameters
    # - clusterGraph: object of class ClusterGraph, to use to pass the clustering
    #                 information.
    # - trainingData: data of training 
    def __init__(self, config, clusterGraph, trainingData = None, sequence_length = None):
            
        KVAE.KalmanVariationalAutoencoder.__init__(self, config, clusterGraph, trainingData, sequence_length)

        return
    
    ###########################################################################
    # FUNCTIONS TO CALCULATE THE LOSSES
    
    # For KVAE_D, we have an additional loss related to the prediction of the
    # odometry values.
    # INPUTS:
    # - actual_params: real value of the odometry,
    # - predicted_params: predicted value of the odometry
    # OUTPUTS:
    # - d_loss: loss that measures the mismatch between real and predicted
    #   odometry. 
    @staticmethod
    def CalculateDMatrixLoss(actual_params, predicted_params):
        
        d_loss = F.mse_loss(actual_params, predicted_params,size_average=False)
        
        return d_loss
    
    # Function to calculate the loss over the predicted odometry, denormalized
    # w.r.t. the extension of the training data.
    # INPUTS:
    # - actual_params: real odometry (normalized)
    # - predicted_params: predicted parameters 
    # - maxXReal,maxYReal,minXReal,minYReal: max and min values over the real odometric data (not normalized)
    @staticmethod
    def CalculateDMatrixLossDenormalized(actual_params, predicted_params, maxXReal = 1, maxYReal = 1, minXReal = 0, minYReal = 0):
        
        # Number of datapoints in the batch
        numberOfElements        = actual_params.shape[0]*actual_params.shape[1]
        
        # Extract the vector of minimum positions in the training dataset
        minRealVector           = torch.zeros(2).to(device)
        minRealVector[0]        = minXReal
        minRealVector[1]        = minYReal
        
        # Extract the vector of differences between min and max in the training dataset
        realDifferenceVector     = torch.zeros(2).to(device)
        realDifferenceVector[0]  = maxXReal - minXReal
        realDifferenceVector[1]  = maxYReal - minYReal
        
        # Extract the vector of min
        minVector     = torch.zeros(2).to(device)
        minVector[0]  = minXReal
        minVector[1]  = minYReal
        
        # Take a version with repetition on the first dimension to subtract
        minVector_rep = torch.unsqueeze(minVector.clone(),0)
        minVector_rep = minVector.repeat(numberOfElements, 1)

        # Denormalize the real values
        actual_params           = actual_params[:,:,0:2]
        actual_params           = torch.reshape(actual_params, [numberOfElements, 2])
        actual_params_denorm    = actual_params*realDifferenceVector 
        actual_params_denorm    = actual_params_denorm + minVector_rep
        
        # Denormalize the predicted values
        predicted_params        = predicted_params[:,:,0:2]
        predicted_params        = torch.reshape(predicted_params, [numberOfElements, 2])
        predicted_params_denorm = predicted_params*realDifferenceVector 
        predicted_params_denorm = predicted_params_denorm + minVector_rep
        
        # Calculate the norm between the real and the predicted parameters
        d_loss_denorm           = torch.linalg.norm(predicted_params_denorm - actual_params_denorm, dim = 1)
        
        # Return the mean over the norm
        return torch.mean(d_loss_denorm)
    
    def CalculateDMatrixLossBestClusterDenormalized(self, actual_params, mu_smooth, alphas, 
                                                    maxXReal = 1, maxYReal = 1, minXReal = 0, minYReal = 0):
        
        # Finding the value of argmax of alpha
        winningClusters   = torch.argmax(alphas, 2)
        
        # Pick the matrices of the winning clusters
        D_winning         = self.kf.D[winningClusters, :, :]
        E_winning         = self.kf.E[winningClusters, :]
        nodesMean_winning = self.kf.nodesMean[winningClusters, :]
        
        predicted_params   = torch.squeeze(torch.matmul(D_winning, mu_smooth))
        predicted_params   = predicted_params + E_winning + nodesMean_winning

        d_loss_denorm = KalmanVariationalAutoencoder_D.CalculateDMatrixLossDenormalized(actual_params, predicted_params, 
                                                                                        maxXReal, maxYReal, minXReal, minYReal)        
        # Return the mean over the norm
        return torch.mean(d_loss_denorm)
    
    @staticmethod
    def CalculateDMatrixLossSeparated(actual_params, predicted_params, alphas):
        
        #d_loss = 0 #F.mse_loss(actual_params, predicted_params,size_average=False)
        
        predicted_params     = predicted_params.permute(2,1,0,3)
                
        real_params_extended = torch.unsqueeze(actual_params, 2)
        real_params_extended = real_params_extended.repeat(1,1,predicted_params.shape[2],1)
        
        d_loss = 0
        for i in range(real_params_extended.shape[0]):
            for j in range(real_params_extended.shape[1]):
                    
                errors_current_instant = torch.abs(real_params_extended[i,j,:,:] - predicted_params[i,j,:,:])
                errors_current_instant = torch.mean(errors_current_instant, axis = 1)
                curr_alpha             = alphas[i, j, :]
                
                d_loss                 += torch.matmul(errors_current_instant.double(),curr_alpha.double())    
                             
        return d_loss
    
    ###########################################################################
    # Smoothing/Filtering functions
        
    def PerformKVAETrainingOverBatchSequence(self, currentImagesBatch,curr_distance,currentClusterParamsBatch):
        
        ########################## KVAE SMOOTHING #############################
        # Training of KVAE is done with call to smoothing function
        reconstructedImagesBatch, a_seq, a_mu, a_var, smooth, A, B, C, alpha_plot, params_pred = \
           self.PerformKVAESmoothingOverBatchSequence(currentImagesBatch,curr_distance)
           
        ################################# LOSSES ##############################
        # Calculate losses
        losses, z_smooth, predicted_params = \
           self.CalculateKVAELoss(a_mu, a_var, reconstructedImagesBatch, currentImagesBatch, 
                                  smooth, A, B, C, currentClusterParamsBatch)
        
        predicted_params = params_pred
        
        return reconstructedImagesBatch, a_seq, a_mu, a_var, losses, \
            z_smooth, smooth, A ,B ,C ,alpha_plot, predicted_params
    
            
    def PerformKVAEFullTrainingOverBatchSequence(self, currentImagesBatch,curr_distance,currentClusterParamsBatch):
        
        ########################## KVAE SMOOTHING #############################
        # Training of KVAE is done with call to smoothing function
        reconstructedImagesBatch, a_seq, a_mu, a_var, smooth, A, B, C, alpha_plot, params_pred = \
           self.PerformKVAESmoothingOverBatchSequence(currentImagesBatch,curr_distance)
           
        ################################# LOSSES ##############################
        # Calculate losses
        losses, z_smooth, predicted_params = \
           self.CalculateKVAELoss(a_mu, a_var, reconstructedImagesBatch, currentImagesBatch, 
                                  smooth, A, B, C, currentClusterParamsBatch)          
        d_loss = KalmanVariationalAutoencoder_D.CalculateDMatrixLoss(currentClusterParamsBatch, params_pred)
        predicted_params = params_pred
        
        return reconstructedImagesBatch, a_seq, a_mu, a_var, losses, \
            z_smooth, smooth, A ,B ,C ,alpha_plot, predicted_params, d_loss
        
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
    # - params_pred: predicted odometry values.
    def PerformKVAESmoothingOverBatchSequence(self, currentImagesBatch,curr_distance):
        
        ################################# VAE #################################
        
        # ENCODE AND DECODE
        reconstructedImagesBatch, a_seq, a_mu, a_var = self.CallVAEEncoderAndDecoderOverBatchSequence(currentImagesBatch)
        
        ################################# KF ##################################
        
        self.kf.initializeCallToKalmanFilter(dist=curr_distance, 
                                             y=a_seq, 
                                             alphaDistProb = self.alphaDistProb)
        
        smooth, A, B, C, alpha_plot, params_preds, mu_preds, Sigma_preds, mu_filts, Sigma_filts = self.kf.smooth()

        return reconstructedImagesBatch, a_seq, a_mu, a_var, smooth, A, B, C, alpha_plot, params_preds, mu_preds, Sigma_preds, mu_filts, Sigma_filts
    
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
    # - params_pred: predicted odometry values.
    def PerformKVAEFilteringOverBatchSequence(self, currentImagesBatch,curr_distance):
        
        ################################# VAE #################################
        
        # ENCODE AND DECODE
        reconstructedImagesBatch, a_seq, a_mu, a_var = self.CallVAEEncoderAndDecoderOverBatchSequence(currentImagesBatch)
        
        ################################# KF ##################################
        
        self.kf.initializeCallToKalmanFilter(dist=curr_distance, 
                                             y=a_seq, 
                                             alphaDistProb = self.alphaDistProb)
     
        mu_pred, Sigma_pred, filter, A, B, C_filter, alpha_values, params_preds = self.kf.filter()
        self.mu_pred    = mu_pred
        self.Sigma_pred = Sigma_pred

        return reconstructedImagesBatch, a_seq, a_mu, a_var, filter, alpha_values, A, B, C_filter, params_preds
                 
    def print(self):
        
        # Print Graph info
        self.clusterGraph.print()         
        # Print VAE info
        self.baseVAE.print()
        
        return