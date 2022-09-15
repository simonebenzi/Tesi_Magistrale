
# This script contains functions for calculating probabilistic distances

import numpy as np
import torch

from ConfigurationFiles import Config_GPU as ConfigGPU

###############################################################################
# ---------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)

###############################################################################
# Multinomial distributions sampling

# Function to performing reparametrization trick to sample on a discrete 
# distribution through gumbel-softmax distribution.
# INPUTS:
# - probabilities: the probabilities of the categorical distribution from 
#   where to sample.
#   This can be either a 1D or a 2D vector. If it is 2D, it must be formed as:
#   (batch size, dimension of categorical distribution)
# - temperature: temperature of the gumbel-softmax distribution. 
#   If we have N classes, and temperature = 0,
#   the sample will be a one-hot distribution (e.g.,  [0 0 1 0 0], if N = 5)
#   If instead the temperature = 1,
#   the sample will be equal to the given probabilities.
# OUTPUTS:
# - sample: sample from the distribution. 
# Example:
# probabilities = torch.zeros(4)
# probabilities[0] = 0.6
# probabilities[1] = 0.3
# probabilities[2] = 0.09
# probabilities[3] = 0.01
# temperature = 0.1
# -> outputs at different calls:
# tensor([9.9997e-01, 2.7528e-05, 3.9385e-08, 5.4083e-08])
# tensor([8.0517e-03, 9.9186e-01, 7.7933e-09, 8.9685e-05])
# tensor([2.7498e-08, 9.9709e-01, 2.8807e-03, 2.6087e-05])
# tensor([9.9970e-01, 1.0388e-10, 2.1651e-07, 2.9521e-04])
# tensor([5.0072e-14, 5.2219e-10, 9.9998e-01, 2.2870e-05])
# Code taken from:
# https://github.com/dev4488/VAE_gumble_softmax
def SampleWithGumbelSoftmaxDistribution(probabilities, temperature, eps=1e-20, div = 10):
    # Sample from the Gumbel distribution
    sg = sample_gumbel(probabilities.size(), eps)
    # Add Gumbel distribution sample to the original probabilities
    y  = probabilities + sg/div #sg/30
    # Softmax
    result = ModifyArrayBasedOnTemperature(y, temperature)
    return result
# Sample from the Gumbel distribution
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

# Sampling from a Multinomial distribution.
# INPUTS:
# - probabilities: probabilities of the multinomial distribution
#   Must be 2D
# OUTPUTS:
# - sampleFromMultinomialFull: e.g. input = probabilities of shape (1,3), output = [0,1,0] -> sampled class 2 of 3
def SampleFromDiscreteDistribution(probabilities):
    
    sampleFromMultinomialFull = torch.zeros(probabilities.shape[0], probabilities.shape[1])
    sampleFromMultinomial     = torch.multinomial(probabilities, 1)
    rows = list(range(0,probabilities.shape[0]))
    sampleFromMultinomialFull[rows,sampleFromMultinomial[:,0]] = 1
    
    return sampleFromMultinomialFull

# Sampling from multinomial distribution
# - probabilities: probabilities of the multinomial distribution
# - dimensionWhereProbabilitiesAre: can be 0 or 1
def SampleClassFromDiscreteDistribution(probabilities, dimensionWhereProbabilitiesAre):
    
    sampleFromMultinomial     = torch.multinomial(probabilities, dimensionWhereProbabilitiesAre)    
    return sampleFromMultinomial

# Sampling from multinomial distribution
# - probabilities: 3D vector. Probabilities on 3-rd dimension!
def SampleClassFromDiscreteDistribution3D(probabilities):
    
    sampleFromMultinomialFinal = torch.zeros(probabilities.shape[0], probabilities.shape[1])
    
    for i in range(probabilities.shape[1]):
        prob= probabilities[:,i,:]
        prob_s = torch.squeeze(prob)
        sampleFromMultinomialFinal[:,i] = torch.squeeze(SampleClassFromDiscreteDistribution(prob_s,1))
    
    return sampleFromMultinomialFinal

###############################################################################
    
def FindHighestValuesAlongDimension(matrix, dimensionWhereToSearchMax):
    
    if type(matrix) == np.ndarray:
        return np.argmax(matrix, dimensionWhereToSearchMax)
    elif type(matrix) == torch.Tensor:
        return torch.argmax(matrix, dimensionWhereToSearchMax)

###############################################################################
# Kullback Leibler

# Find Kullback Leibler distance between two gaussian distributions
# INPUTS:
# pm: mean of distribution p;
# pv: variance of distribution p;
# qm: mean of distribution q;
# qv: variance of distribution q;
def gau_kl(pm, pv, qm, qv):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod(axis)
    # Inverse of diagonal covariance qv
    iqv = 1./qv
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
             + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
             + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm)))                     # - N
    
# Same as 'gau_kl', but in torch.
def gau_klTorch(pm, pv, qm, qv):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod(axis)
    # Inverse of diagonal covariance qv
    iqv = 1./qv
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (torch.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
             + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
             + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm)))                     # - N

# Kullback leibler calculation
def KLDiv(p, q):
    """Kullback-Liebler divergence from multinomial p to multinomial q,
    expressed in nats."""
    if (len(q.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Clip before taking logarithm to avoid NaNs (but still exclude
    # zero-probability mixtures from the calculation)
    return (p * (np.log(p.clip(1e-10,1))
                 - np.log(q.clip(1e-10,1)))).sum(axis)   # - N
    
# Kullback leibler calculation inside MJPF
def single_KLD_Abnormality(PP, QQ, KLDAbnMax, histogramProb, N):
    
    PP = PP.cpu()
    QQ = QQ.cpu()
    
    if torch.sum(torch.isinf(QQ)) >= 1:
        KLD_simmetrica = KLDAbnMax 
    elif torch.sum(torch.isnan(QQ)) >= 1:
        KLD_simmetrica = KLDAbnMax
    elif torch.sum(torch.isnan(PP)) >= 1:
        KLD_simmetrica = KLDAbnMax
    else:
        KLD_simmetrica = (histogramProb/N)*KLDiv(PP,QQ) + (histogramProb/N)*KLDiv(QQ,PP) #to achieve symmerty
    
    if torch.sum(torch.isinf(KLD_simmetrica)) >= 1:
        KLD_simmetrica = KLDAbnMax;
    
    return KLD_simmetrica

## Input:
# totNumOfSuperstates: total number of Superstates
# N: total number of Particles
# histogram: histogram at time t-1 (after PF resampling)
# transitionMat: the transition matrix learned from previous experience
# probability_lamdaS: probability vector representing a discrete probability disctribution         
def KLD_Abnormality(nSuperstates, N, histogram, transitionMat, probability_lamdaS, KLDAbnMax):
    
    ##Procedure:
    sommaKLD_simmetrica = 0
    
    for indKLD in range(nSuperstates):
        particella = histogram[indKLD]
        
        if particella > 0:
            
            PP = torch.squeeze(transitionMat[indKLD,:]) +1e-20 # add 1e-100 since KLD doesnt allow zero values
            QQ = torch.squeeze(probability_lamdaS)
            
            KLD_simmetrica = single_KLD_Abnormality(PP, QQ, KLDAbnMax, particella, N)
            
            sommaKLD_simmetrica = sommaKLD_simmetrica + KLD_simmetrica
        
    return sommaKLD_simmetrica

###############################################################################
# Bhattacharya distance

# Find Bhattacharyya distance between two gaussian distributions
# INPUTS:
# pm: mean of distribution p;
# pv: variance of distribution p;
# qm: mean of distribution q;
# qv: variance of distribution q;
def CalculateBhattacharyyaDistance(pm, pv, qm, qv):
    
    # Copyright (c) 2008 Carnegie Mellon University
    #
    # You may copy and modify this freely under the same terms as
    # Sphinx-III
    
    #__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu>"
    #__version__ = "$Revision$"

    """
    Classification-based Bhattacharyya distance between two Gaussians
    with diagonal covariance.  Also computes Bhattacharyya distance
    between a single Gaussian pm,pv and a set of Gaussians qm,qv.
    Does not work for calculation from a set of gaussians to another
    set of gaussians. Unfortunately, in this case, a 'for' loop must
    be used.
    """
    
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Difference between means pm, qm
    diff = qm - pm
    # Interpolated variances
    pqv = (pv + qv) / 2.
    # Log-determinants of pv, qv
    
    ldpv = np.log(pv).sum()
    ldqv = np.log(qv).sum(axis)
    # Log-determinant of pqv
    ldpqv = np.log(pqv).sum(axis)
    # "Shape" component (based on covariances only)
    # 0.5 log(|\Sigma_{pq}| / sqrt(\Sigma_p * \Sigma_q)
    norm = 0.5 * (ldpqv - 0.5 * (ldpv + ldqv))
    # "Divergence" component (actually just scaled Mahalanobis distance)
    # 0.125 (\mu_q - \mu_p)^T \Sigma_{pq}^{-1} (\mu_q - \mu_p)
    dist = 0.125 * (diff * (1./pqv) * diff).sum(axis)

    return dist + norm

# Bhattacharya distance with torch between two arrays of gaussians.
def CalculateBhattacharyyaDistanceBetweenTwoArraysOfGaussians(mean_p, cov_p, mean_q, cov_q):
    
    # Calculate the loss for each value
    bhattacharya_distance_sum = 0
    for j in range(mean_p.shape[0]):
        
        # For current value
        bhattacharya_distance_current = CalculateBhattacharyyaDistanceTorch(
            mean_p[j], torch.diagonal(cov_p[j], dim1 =0, dim2 = 1),
            mean_q[j], torch.diagonal(cov_q[j], dim1 =0, dim2 = 1)) 
        
        # Add to sum
        bhattacharya_distance_sum += bhattacharya_distance_current   
    
    return bhattacharya_distance_sum

# Same as 'CalculateBhattacharyyaDistance', but in torch.
def CalculateBhattacharyyaDistanceTorch(pm, pv, qm, qv):
    
    # Copyright (c) 2008 Carnegie Mellon University
    #
    # You may copy and modify this freely under the same terms as
    # Sphinx-III
    
    #__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu>"
    #__version__ = "$Revision$"

    """
    Classification-based Bhattacharyya distance between two Gaussians
    with diagonal covariance.  Also computes Bhattacharyya distance
    between a single Gaussian pm,pv and a set of Gaussians qm,qv.
    Does not work for calculation from a set of gaussians to another
    set of gaussians. Unfortunately, in this case, a 'for' loop must
    be used.
    """
    
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0

    # Difference between means pm, qm
    diff = qm - pm
    # Interpolated variances
    pqv = (pv + qv) / 2.
    # Log-determinants of pv, qv
    
    ldpv  = torch.log(pv).sum()
    ldqv  = torch.log(qv).sum(axis)
    # Log -determinant of pqv
    ldpqv = torch.log(pqv).sum(axis)
    # "Shape" component (based on covariances only)
    # 0.5 log(|\Sigma_{pq}| / sqrt(\Sigma_p * \Sigma_q)
    norm = 0.5 * (ldpqv - 0.5 * (ldpv + ldqv))
    # "Divergence" component (actually just scaled Mahalanobis distance)
    # 0.125 (\mu_q - \mu_p)^T \Sigma_{pq}^{-1} (\mu_q - \mu_p)
    dist = 0.125 * (diff * (1./pqv) * diff).sum(axis)
    
    return dist + norm  

###############################################################################
# Softmax
    
# Finding softmax cross entropy loss
def softmax_cross_entropy_with_softtarget(input, target, reduction='mean'):
        
        logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
        batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
        
        if reduction == 'none':
            return batchloss
        elif reduction == 'mean':
            return torch.mean(batchloss)
        elif reduction == 'sum':
            return torch.sum(batchloss)
        else:
            raise NotImplementedError('Unsupported reduction mode.')
    
# Testing softmax
def testOf_softmax_cross_entropy_with_softtarget():
    
    loss = torch.nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target1 = torch.empty(3, dtype=torch.long).random_(5)
    target1[0] = 1
    target1[1] = 0
    target1[2] = 2
    output1 = loss(input, target1)
    
    loss = torch.nn.CrossEntropyLoss()
    target2 = torch.zeros(3, 5, dtype=torch.long)
    target2[0,1] = 1
    target2[1,0] = 1
    target2[2,2] = 1
    output2 = softmax_cross_entropy_with_softtarget(input, target2)
    
    return

###############################################################################
# Perpendicular/parallel distances

# Finding parallel and perpendicular distance between two clusters
def CalculateParallelAndPerpendicularDistanceBetweenClustersBothDirections(mean_a, mean_b):
    
    # Treating A as the point
    parallelDistanceA, perpendicularDistanceA = CalculateParallelAndPerpendicularDistanceToCluster(mean_a, mean_b, mean_a)
    # Treating B as the point
    parallelDistanceB, perpendicularDistanceB = CalculateParallelAndPerpendicularDistanceToCluster(mean_a, mean_b, mean_b)
    
    return parallelDistanceA, perpendicularDistanceA, parallelDistanceB, perpendicularDistanceB

def CalculateParallelAndPerpendicularDistanceToCluster(mean_a, mean_b, meanChosenCluster):
    
    # Calculate the difference between the two points along all directions
    innovationAB         = mean_b - mean_a
    # Consider only the position error
    positionInnovationAB = innovationAB[0:2]
    # Velocity of cluster taken as point
    velocityCluster      = meanChosenCluster[2:4]
    # The distance vector between A and B is projected along the velocity
    # of the cluster. In this way, we find the distance vector along the 
    # parallel.        
    X_projected = projectPointOnLine2D(point = positionInnovationAB, line = velocityCluster)
    
    # Parallel distance
    parallelDistance      = np.linalg.norm(X_projected)
    # Perpendicular distance
    perpendicularDistance = np.linalg.norm(X_projected - positionInnovationAB)
    
    return parallelDistance, perpendicularDistance

# Function for projecting a point on a line
def projectPointOnLine2D(point, line):
    
    dot_AP_AB   = np.dot(line, point)
    dot_AB_AB   = np.dot(line, line)
    AB_line     = line
    projection = dot_AP_AB/ dot_AB_AB*AB_line
    
    return projection

###############################################################################
# Calculating probabilities/modifying arrays using tempered distances

# Finding a vector of probabilities given a vector of distances.
# Values having higher distance will have lower probability.
# This function can be used to find the probability of clusters given the
# distance of a point from each of them.
# INPUTS:
# - distances
# - temperature: temperature value. The higher, the more peaked the 
#   result will be around the most probable cluster.
# OUTPUTS:
# - probabilities: vector of probabilities.
def CalculateProbabilitiesFromDistances(distances, temperature, jumpTemperature = 0.25):
    
    dimensions = distances.ndim
    
    if dimensions == 1:
        distances = torch.unsqueeze(distances, 0)
        
    p_distance    = torch.zeros(distances.shape[0], distances.shape[1])
    temperatures  = torch.zeros(distances.shape[0])
    
    # Elevate the distances to the temperature
    pow_distances_temperature = torch.pow(distances,temperature)
    # Inverse
    p_distance                = 1./(pow_distances_temperature + 1e-60)    

    for i in range(distances.shape[0]):
        
        working = False
        
        # Values for current batch point
        temperature_curr = temperature       
        distances_curr   = distances[i,:].clone()

        while(working == False):
    
            # Elevate the distances to the temperature
            pow_distances_temperature_curr = torch.pow(distances_curr,temperature_curr)
            # Inverse
            p_distance_curr                = 1./(pow_distances_temperature_curr + 1e-60)
            
            if torch.isnan(p_distance_curr).all() or (p_distance_curr == 0.).all():
                temperature_curr -= jumpTemperature
            else:
                working = True    
                p_distance[i,:] = p_distance_curr.clone()
                temperatures[i] = temperature
 
    # Sum of inverse
    sum_inverse               = torch.sum(p_distance, axis=1)
    # Expand on second dimension
    sum_inverse_expanded      = torch.unsqueeze(sum_inverse, axis=1)
    # Get probabilities
    probabilities             = torch.div(p_distance, sum_inverse_expanded) 
    
    if dimensions == 1:
        probabilities = torch.squeeze(probabilities)
    
    return probabilities

# In this function, after calculating the probabilities from the distances, 
# we sample from the found probabilities and switch the sampled probability
# with the highest one.
def CalculateProbabilitiesFromDistancesRandomizing(distances, temperature, jumpTemperature = 0.25):
    
    temperature_base = 50
    
    alphas, temperatures, p_distance = CalculateProbabilitiesFromDistances(distances, temperature_base)
    
    # Argmax
    argmaxes_alphas                                  = torch.argmax(alphas, dim = 1)
    # Sample
    sampleFromMultinomialFull, sampleFromMultinomial = SampleFromDiscreteDistribution(alphas)
    sampleFromMultinomial                            = torch.squeeze(sampleFromMultinomial)
    
    # Switch
    selection_range                                             = np.arange(0, distances.shape[0])
    distances_temporary                                         = distances.clone()
    distances_temporary[selection_range, argmaxes_alphas]       = distances[selection_range, sampleFromMultinomial].clone()
    distances_temporary[selection_range, sampleFromMultinomial] = distances[selection_range, argmaxes_alphas].clone()
    
    # Calculate alphas again
    alphas_new, temperatures_new, p_distance_new = CalculateProbabilitiesFromDistances(
        distances_temporary, temperature)
    
    return alphas_new

# Function to modify an array to be more peaked around the highest value.
# INPUTS:
# - array: the array to modify,
# - temperature: temperature value. The higher, the more peaked the 
#   result will be around the highest value.
# OUTPUTS:
# - temperedArray: modified array.
def ModifyArrayBasedOnTemperature(array, temperature):
    
    dimensions = array.ndim
    
    if dimensions == 1:
        array = torch.unsqueeze(array, 0)
    
    # Elevate the distances to the temperature
    pow_array_temperature     = torch.pow(array,temperature)
    # Sum of elevated values
    sum_elevated_array        = torch.sum(pow_array_temperature, axis=1)
    # Expand on second dimension
    sum_elevated_expanded     = torch.unsqueeze(sum_elevated_array, axis=1)
    # Get probabilities
    temperedArray             = torch.div(pow_array_temperature, sum_elevated_expanded) 
    
    if dimensions == 1:
        temperedArray = torch.squeeze(temperedArray)
    
    return temperedArray

