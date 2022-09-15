# -*- coding: utf-8 -*-
"""
Created on Mon May  3 15:39:28 2021

@author: asus
"""

import torch
from ConfigurationFiles import Config_GPU as ConfigGPU 

###############################################################################
# ----------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)

###############################################################################
# --------------------------  GeometryHandler   -------------------------------
###############################################################################

# STATIC Class to Handle geometry
class GeometryHandler(object):
    
    @staticmethod
    #Function to project a point on a 2D plane
    def ProjectPointOnPlane2D(velocity, orthogonal_to_cluster_mean):
         
        orthogonal_to_cluster_mean = torch.squeeze(orthogonal_to_cluster_mean)
        
        # what is the dimension of velocity vector
        dimensions  = velocity.shape[0] 
        A_point     = torch.zeros(dimensions).to(device)
        X_projected = torch.zeros(dimensions).to(device)
                
        dot_AP_AB     = torch.dot(orthogonal_to_cluster_mean, velocity)
        dot_AB_AB     = torch.dot(orthogonal_to_cluster_mean, orthogonal_to_cluster_mean)
        AB_line       = orthogonal_to_cluster_mean
        
        X_projected = A_point + torch.div(dot_AP_AB,dot_AB_AB)*AB_line
        
        return X_projected