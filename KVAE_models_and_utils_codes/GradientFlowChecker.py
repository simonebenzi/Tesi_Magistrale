
# This is a class that contains a lot of functions useful for performing
# checking of how the gradient is flowing inside a network.

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D 

from ConfigurationFiles import Config_GPU as ConfigGPU

###############################################################################
# ---------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)

###############################################################################

def plot_and_save_grad_flow(named_parameters, fileName):
    
    layers, ave_grads, max_grads = calculateGrads(named_parameters)
    
    plotGrads(layers, max_grads)
    plt.savefig(fileName + '_max.png')
    
    plotGrads(layers, ave_grads)
    plt.savefig(fileName + '_average.png')
    
    return

def plot_and_save_grad_flow_together(named_parameters, fileName):
    
    layers, ave_grads, max_grads = calculateGrads(named_parameters)
    plotGradsTogether(layers, max_grads, ave_grads)
    plt.savefig(fileName)
    
    return

def plotGradsTogether(layers, max_grads, ave_grads):
    
    max_grad_value = np.asarray(max_grads).max()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=max_grad_value) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    return

def plotGrads(layers, grads):
    
    max_grad_value = np.asarray(grads).max()
    plt.bar(np.arange(len(grads)), grads, alpha=0.1, lw=1, color="c")
    plt.hlines(0, 0, len(grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(grads))
    plt.ylim(bottom = -0.001, top=max_grad_value) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['grads', 'zero-gradient'])
    
    return

def calculateGrads(named_parameters):
    
    ave_grads = []
    max_grads= []
    layers = []
    
    plt.close('all')
    
    for n, p in named_parameters:

        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    
    return layers, ave_grads, max_grads