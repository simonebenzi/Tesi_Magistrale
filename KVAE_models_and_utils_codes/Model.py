# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:39:03 2021

@author: giulia.slavic
"""

# Abstract class from which torch Neural Network modules such as VAE and KVAE should
# inherit.

import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from ConfigurationFiles import Config_GPU as ConfigGPU

###############################################################################
# ---------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)

###############################################################################
# -------------------------------  MODEL   ------------------------------------
###############################################################################

class Model(nn.Module, ABC):
    
    # Function to load a trained model on the GPU (if available) or on the 
    # cpu if not available.
    # INPUTS:
    # - loadFile: path to the file from where the model should be loaded.
    def LoadTrainedModel(self, loadFile):
        
        if device.type == 'cpu': # CPU case
            self.load_state_dict(torch.load(loadFile,map_location=torch.device('cpu')))
        else: # GPU case
            self.load_state_dict(torch.load(loadFile))
            
        self.to(device) # bring to CPU or GPU
        
        return self
    