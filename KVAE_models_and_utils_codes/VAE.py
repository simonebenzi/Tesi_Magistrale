
# This script contains functions for the VAE CLASS
# This CLASS allows to create a VAE and perform encoding and decoding.

###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import numpy as np
import os

from torchsummary import summary
from torchvision.utils import save_image

from ConfigurationFiles import Config_GPU as ConfigGPU
from KVAE_models_and_utils_codes import Model
from KVAE_models_and_utils_codes import SummaryHolder                   as SH
from KVAE_models_and_utils_codes import SummaryHolderLossesAcrossEpochs as SHLAE

###############################################################################
# ---------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)

###############################################################################
# ------------------       VariationalAutoencoder   --------------------------
###############################################################################

###############################################################################
# Summary objects: losses names

# This contains only those value of which we are interested in the MEAN over all epochs
summaryNamesAllEpochsVAE    = ['VAE_losses', 'Reconstruction_losses', 'KLD_losses', 'Total_loss']
# This instead, contains values that are only epoch specific
summaryNamesCurrentEpochVAE = summaryNamesAllEpochsVAE + ['a_states', 'learning_rates']

###############################################################################
# Exception classes

class TooManyConvolutionalLayersException(Exception):

    def __init__(self, dimensionX, dimensionY):
        self.dimensionX = dimensionX
        self.dimensionY = dimensionY    
    def __str__(self):
        message = 'The image dimensions at the end of the convolution layers are:' + \
                  ' ' + str(self.dimensionX) + ' and' + \
                  ' ' + str(self.dimensionY) + ',' + \
                  ' which are < 1.'                 
        return message

###############################################################################
# VAE class

class VAE(Model.Model):
    
    @staticmethod
    def ReturnSummaryNamesAllEpochsVAE():
        
        # This contains only those value of which we are interested in the MEAN over all epochs
        summaryNamesAllEpochsVAE    = ['VAE_losses', 'Reconstruction_losses', 'KLD_losses', 'Total_loss']
        
        return summaryNamesAllEpochsVAE
    
    @staticmethod
    def ReturnSummaryNamesCurrentEpochVAE():
        
        # Summary names over all epochs
        summaryNamesAllEpochsVAE    = VAE.ReturnSummaryNamesAllEpochsVAE()
        # This instead, contains values that are only epoch specific
        summaryNamesCurrentEpochVAE = summaryNamesAllEpochsVAE + ['a_states', 'learning_rates']
        
        return summaryNamesCurrentEpochVAE
    
    @staticmethod
    def ReturnBothSummaryNamesTypes():
        
        # Summary names over all epochs
        summaryNamesAllEpochsVAE    = VAE.ReturnSummaryNamesAllEpochsVAE()
        # This instead, contains values that are only epoch specific
        summaryNamesCurrentEpochVAE = summaryNamesAllEpochsVAE + ['a_states', 'learning_rates']
        
        return summaryNamesAllEpochsVAE, summaryNamesCurrentEpochVAE
    
    # Function to extract the dimensions of each layer of the ENCODER
    # using the formula:
    # U = (I - K + 2*P)/2 + 1
    # where:
    # - U is the output dimension (= layersDimensions[i, :])
    # - I is the input dimension (curr_size)
    # - K is the kernel dimension (self.kernel)
    # - P is the padding (set to 0)
    # - S is the stride (self.stride)
    # Each of them has 2 parts, for X and Y
    # Input: VAE and its parameters set in Config file.
    # Output: implicitly, dimensions of each layer of the encoder (layersDimensions)
    def FindEncoderDimensions(self):
        
        # Array to contain the dimensions of the encoder
        self.layersDimensions = np.zeros((self.num_filters + 1, 2))
        
        # We take as irst input dimension the dimensions of the images
        curr_size = np.asarray(self.image_size)
        
        # Finding all the layer dimensions
        for i in range(self.num_filters + 1):
            
            # Input dimension
            self.layersDimensions[i, :] = curr_size
            # Find output dimension
            # U = (I - K + 2*P)/2 + 1,    with P = 0
            curr_size                   = np.floor((curr_size - self.kernel)/self.stride + 1)
            
            
        return
            
    # Find the paddings for the DECODER to use to keep same layers dimensions 
    # in encoder and decoder. Also kernel values could change w.r.t. to initial 
    # indication due to presence of even dimensions (typically in last deconvolution)
    # First we find the optimum padding with formula:
    # P = (S*(I-1) + K - U)/2
    # where:
    # - U is the output dimension (currEncoderLayerDimensionOut)
    # - I is the input dimension (currDecoderLayerDimensionIn)
    # - K is the kernel dimension (self.kernel)
    # - P is the padding (paddings[i, :])
    # - S is the stride (self.stride)
    # We take the ceil value of it.
    # Then we find the optimal value of kernel. If padding was sufficient to have
    # same sizes, the kernel will remain the one set in the Config file. Otherwise
    # it will change. The formula used to calculate it is:
    # K = U + 2*P - S*(I-1)
    # Input: VAE and its parameters set in Config file.
    # Output: implicitly, paddings and kernels to use in the DECODER, i.e., 
    #         self.paddings and self.kernels, respectively.
    def FindPaddingsAndKernels(self):
        
        # Paddings to use in decoder
        self.paddings = np.zeros((self.num_filters, 2))
        # Kernels to use in decoder
        self.kernels  = np.zeros((self.num_filters, 2))
        
        for i in range(self.num_filters):
            
            # Current nput dimension
            currDecoderLayerDimensionIn   = self.layersDimensions[self.num_filters - i]
            # Current output dimension
            currEncoderLayerDimensionOut  = self.layersDimensions[self.num_filters - i - 1]
            # Find padding:
            # P = (S*(I-1) + K - U)/2
            self.paddings[i, :]           = (self.stride*(currDecoderLayerDimensionIn - 1) + self.kernel - currEncoderLayerDimensionOut)/2
            self.paddings[i, :]           = np.abs(np.ceil(self.paddings[i, :]))
            # Find kernel:
            # K = U + 2*P - S*(I-1)
            self.kernels[i, :]            = currEncoderLayerDimensionOut + 2*self.paddings[i, :] - self.stride*(currDecoderLayerDimensionIn - 1)
            self.kernels[i, :]            = np.abs(np.ceil(self.kernels[i, :]))

        return
    
    # Function to build the ENCODER CONVOLUTIONAL LAYERS
    # The layers are built by using:
    # - the same stride defined in self.stride
    # - the kernel defined in self.kernel
    # - dimension of filters as defined in self.dim_filters
    # - zero padding
    # - putting a LeakyReLU after each covolutional layer
    # Implicit output: self.encoder -> encoder function containing all the 
    #                  encoder layers.
    def DefineEncoder(self):
        
        # Where to put list of layers
        encoderLayers = []
        
        # ENCODER LAYERS
        for i in range(self.num_filters):
            
            # Number of input channels of current layer
            if i == 0:
                # If this is the first layer, the number of input channels
                # is the number of image channels...
                input_channels = self.image_channels
            else:
                # ... otherwise, we use the number of filters per layer
                input_channels = self.dim_filters[i-1]
            
            # Number of output channels of current layer
            output_channels    = self.dim_filters[i]
            
            # Define the current layer
            currentLayer       = nn.Sequential(*[
                                                 nn.Conv2d(in_channels  = input_channels,
                                                           out_channels = output_channels,
                                                           kernel_size  = (self.kernel, self.kernel),
                                                           padding      =  0, 
                                                           stride       = (self.stride    , self.stride)),
                                                 nn.BatchNorm2d(output_channels),
                                                 nn.LeakyReLU(0.1) 
                                                 ])
            
            # Add layer to the list
            encoderLayers.append(currentLayer)
            
        # Final encoder
        self.encoder = nn.Sequential(*encoderLayers)
            
        return
        
    # Function to build the DECODER CONVOLUTIONAL LAYERS
    # The layers are built by using:
    # - the same stride defined in self.stride
    # - the kernels defined in self.kernels
    # - dimension of filters as defined in self.dim_filters
    # - the paddings defined as in self.paddings
    # - putting a LeakyReLU after each covolutional layer
    # Implicit output: self.decoder -> decoder function containing all the 
    #                  decoder layers.
    def DefineDecoder(self):
        
        # Where to put list of layers
        decoderLayers = []
        
        # DECODER LAYERS
        for i in range(self.num_filters):
            
            # Number of input channels of current layer
            if i == self.num_filters - 1:
                # If this is the last layer, the number of input channels
                # is the number of image channels...
                output_channels = self.image_channels
            else:
                # ... otherwise, we use the number of filters per layer, 
                # with inverse order
                output_channels = self.dim_filters[self.num_filters - i - 2]
            
            # Number of output channels of current layer
            input_channels      = self.dim_filters[self.num_filters - i - 1]
            
            # Define the current layer
            currentLayer        = nn.Sequential(*[
                                                  nn.ConvTranspose2d(in_channels  = input_channels,
                                                                     out_channels = output_channels,
                                                                     kernel_size  = (int(self.kernels[i,0])  , int(self.kernels[i,1])),
                                                                     padding      = (int(self.paddings[i,0]) , int(self.paddings[i,1])),
                                                                     stride       = (self.stride        , self.stride)),
                                                  nn.BatchNorm2d(output_channels),
                                                  ])
            
            # Add layer to the list                  
            decoderLayers.append(currentLayer)#cuda())
            
            # Add Relu to the list, unless i is the last layer of the decoder
            if i != self.num_filters - 1:
                reluLayer = nn.LeakyReLU(0.1)
                decoderLayers.append(reluLayer)
            
        # Final decoder
        self.decoder = nn.Sequential(*decoderLayers)
        
        return
    
    # Function to define the Fully Connected Layers
    def DefineFullyConnectedLayers(self):
        
        # Layer to mu/a in encoder
        self.fc1 = nn.Linear(self.finalDimension, self.z_dim)
        # Layer to sigma in encoder
        self.fc2 = nn.Linear(self.finalDimension, self.z_dim)
        
        # Layer from z/a in decoder
        self.fc3 = nn.Linear(self.z_dim, self.finalDimension)
        
        return
    
    # Build the VAE with the given information
    def BuildVAE(self):
        
        # Encoder (except fully connected layers)
        self.DefineEncoder()
        # Decoder (except fully connected layers)
        self.DefineDecoder()
        # Fully connected layers of encoder and decoder
        self.DefineFullyConnectedLayers()
        
        return
    
    # Checking for wrong VAE structure
    def CheckForExceptionsOnStructure(self):
        
        # Check that the first two dimensions at the end of convolutions did not go below zero
        if self.lastHiddenDimensions[0] < 1 or self.lastHiddenDimensions[1] < 1:
            raise TooManyConvolutionalLayersException(self.lastHiddenDimensions[0], 
                                                      self.lastHiddenDimensions[1])
        return
        
    # Initialize the network
    def __init__(self, z_dim, image_channels, image_size, dim_filters, kernel, stride):
        super(VAE, self).__init__()
        
        # From indications set in the configuration file
        self.z_dim          = z_dim
        self.image_channels = image_channels
        self.image_size     = image_size
        self.dim_filters    = dim_filters
        self.num_filters    = len(dim_filters)
        self.kernel         = kernel
        self.stride         = stride
        
        # Find paddings and final kernel values
        self.FindEncoderDimensions()
        self.FindPaddingsAndKernels()
        
        # Dimension of last filter (to take from Config)
        # e.g., if dim_filters = [32, 64, 128], 
        # num_filters = 128
        self.lastFilterDimension  = self.dim_filters[self.num_filters - 1]
        
        # Final hidden dimension
        self.lastHiddenDimensions = self.layersDimensions[self.num_filters, :]
        
        self.finalDimension       = self.lastHiddenDimensions[0]*self.lastHiddenDimensions[1]
        self.finalDimension       = self.finalDimension*self.lastFilterDimension
        self.finalDimension       = int(self.finalDimension)
        
        # Check for exceptions on VAE structure
        self.CheckForExceptionsOnStructure()
        
        # Build the VAE
        self.BuildVAE()
        
        return
    
    # Print function
    def print(self):
        
        print('Dimension of latent state:')
        print(self.z_dim)
        print('Number of image channels:')
        print(self.image_channels)
        print('Image size:')
        print(self.image_size)
        print('Filter dimensions:')
        print(self.dim_filters)
        print('Number of filters:')
        print(self.num_filters)
        print('Base dimensions of kernels:')
        print(self.kernel)
        print('Stride:')
        print(self.stride)
        print('Dimensions of encoder:')
        print(self.layersDimensions)
        print('Last Convolution dimensions (as from above):')
        print(self.lastHiddenDimensions)
        print('Paddings of decoder:')
        print(self.paddings)
        print('Kernels of decoder:')
        print(self.kernels)
        print('Final dimension after convolutions and before FC layers:')
        print(self.finalDimension)
        
        return
    
    # Function to show all the layers of the VAE
    def PrintVAELayers(self, image_channels):
        
        summary(self, input_size=(image_channels,self.image_size[0],self.image_size[1]), device=device.type)
        
        return
                                              
    # Flatten the features along a single dimension (plus batch size)
    # OUTPUT of Flatten -> [batch_size; remaining_dimensions]
    # To use at end of ENCODER
    def Flatten(self, input):
        
        return input.view(input.size(0),-1)
    
    # Unflatten the features from the single dimensions to the original ones
    # before flattening.
    # INPUT of Unflatten -> [batch_size; remaining_dimensions]
    # To use at the beginning of DECODER.
    def Unflatten(self, input):
        
        return input.view(-1, self.lastFilterDimension, 
                              int(self.lastHiddenDimensions[0]), 
                              int(self.lastHiddenDimensions[1]))
        
    # Reparametrization function
    # Inputs: - mu: latent state
    #         - logvar: covariance
    # Output: - z: sampled latent state
    def Reparameterize(self, mu, logvar):
        
        std = logvar.mul(0.5).exp_().to(device)
        esp = torch.randn(*mu.size()).to(device)
        
        z   = mu + std * esp
                
        return z
    
    # Bottleneck
    # Input:  - h: input of bottleneck, which is output of fully connected layer
    #           of encoder
    # Output: - mu: latent state
    #         - logvar: covariance
    #         - z: sampled latent state
    def bottleneck(self, h):
        
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.Reparameterize(mu, logvar)

        return z, mu, logvar
    
    # Encode image to latent space
    # Input:  - x: image
    # Output: - mu: latent state
    #         - logvar: covariance
    #         - z: sampled latent state
    def Encode(self, x):
        
        # Go through VAE's encoder
        x = self.encoder(x)    
        # Flatten
        h = self.Flatten(x)
        
        # Bottleneck of the VAE
        z, mu, logvar = self.bottleneck(h)

        return z, mu, logvar
    
    # Decode sampled latent state to image
    # Input:  - z: sampled latent state
    # Output: - x: reconstructed image
    def Decode(self, z):
        
        # Unflatten
        z = self.fc3(z)
        z = self.Unflatten(z)
        
        # Go through VAE's decoder
        z = self.decoder(z)
        
        # Sigmoid function
        x = F.sigmoid(z)

        return x
    
    # Forward function: from an image, performs encoding and decoding.
    # Input:   - x: image
    # Outputs: - x_rec: reconstructed image
    #          - mu: latent state
    #          - logvar: covariance
    #          - z: sampled latent state
    def forward(self, x):
        
        # Encoding
        z, mu, logvar  = self.Encode(x)
        # Decoding
        x_rec = self.Decode(z)
        
        return x_rec, z, mu, logvar
    
    @staticmethod
    def ObtainVarFromLogvar(a_var):
        
        a_var_exp      = torch.exp(a_var)
        
        return a_var_exp
    
    @staticmethod
    def ObtainCovFromLogvar(a_var):
        
        a_var_exp      = VAE.ObtainVarFromLogvar(a_var)
        a_var_exp_cov  = torch.diag_embed(a_var_exp)

        return a_var_exp_cov
        
        
    
    
    ###########################################################################
    # These functions are for debugging, to see for example if the parameters
    # of VAE have been changing from one optimization step to another one etc.
    
    # Getting the weights as output
    
    # Fully connected layers
    def GetParametersOfFullyConnectedLayerFc1(self):
        return self.fc1.weight.detach().cpu().numpy()
    def GetParametersOfFullyConnectedLayerFc2(self): 
        return self.fc2.weight.detach().cpu().numpy()
    def GetParametersOfFullyConnectedLayerFc3(self):
        return self.fc3.weight.detach().cpu().numpy()
    # Encoder
    def GetParametersOfFirstEncoderLayer(self):
        return self.encoder[0][0].weight.detach().cpu().numpy()
    # Decoder
    def GetParametersOfFirstDecoderLayer(self):
        return self.decoder[0][0].weight.detach().cpu().numpy()
    
    # Printing the weights
    
    # Fully connected layers
    def PrintParametersOfFullyConnectedLayerFc1(self):
        print(self.GetParametersOfFullyConnectedLayerFc1())
        return 
    def PrintParametersOfFullyConnectedLayerFc2(self): 
        print(self.GetParametersOfFullyConnectedLayerFc2())
        return
    def PrintParametersOfFullyConnectedLayerFc3(self):
        print(self.GetParametersOfFullyConnectedLayerFc3())
        return
    # Encoder
    def PrintParametersOfFirstEncoderLayer(self):
        print(self.GetParametersOfFirstEncoderLayer())
        return
    # Decoder
    def PrintParametersOfFirstDecoderLayer(self):
        print(self.GetParametersOfFirstDecoderLayer())
        return
    
    # Comparing previous parameters with current ones
    
    # General
    @staticmethod
    def CompareParameters(previousParameters, currentParameters):
        return(previousParameters == currentParameters).all()
    # Fully connected layers
    def CompareParametersOfFullyConnectedLayerFc1(self,previousParameters):
        currentParameters = self.GetParametersOfFullyConnectedLayerFc1()
        return VAE.CompareParameters(currentParameters, previousParameters)
    def CompareParametersOfFullyConnectedLayerFc2(self,previousParameters):
        currentParameters = self.GetParametersOfFullyConnectedLayerFc2()
        return VAE.CompareParameters(currentParameters, previousParameters)
    def CompareParametersOfFullyConnectedLayerFc3(self,previousParameters):
        currentParameters = self.GetParametersOfFullyConnectedLayerFc3()
        return VAE.CompareParameters(currentParameters, previousParameters)
    # Encoder
    def CompareParametersOfFirstEncoderLayer(self,previousParameters):
        currentParameters = self.GetParametersOfFirstEncoderLayer()
        return VAE.CompareParameters(currentParameters, previousParameters)
    # Decoder
    def CompareParametersOfFirstDecoderLayer(self,previousParameters):
        currentParameters = self.GetParametersOfFirstDecoderLayer()
        return VAE.CompareParameters(currentParameters, previousParameters)
    
    # Function to compare if the parameters of a given VAE are the same as the
    # ones of the current VAE. 
    # This applies for ALL parameters of convolutional and FC layers.
    # !! However, this does not consider the statistics of the batchnorm 
    # layers. 
    # !! In case the other VAE is the same VAE at a previous epoch, 
    # remember to have it saved as a deep copy and not a pointer, or it 
    # will simply correspond to the current one, i.e., use:
    # previousVAE = copy.deepcopy(kvae.baseVAE)
    def CompareVAEparamsWithCurrentOne(self, VAEtoCompare):
        
        for p1, p2 in zip(VAEtoCompare.parameters(), self.parameters()):
            if p1.data.ne(p2.data).sum() > 0:
                return False # they are different
            
        return True # they are equal
    
    ###########################################################################
    # Functions for freezing layers
    
    # When fine-tuning or performing similar operations, remember to freeze
    # the batchnorm layers: it the statistics of the data are quite different, 
    # the batchnorm layers statistics would be recalculated and the network
    # will not work any more. So, to avoid this, freeze these layers.
    # See also:
    # https://stackoverflow.com/questions/63016740/why-its-necessary-to-frozen-all-inner-state-of-a-batch-normalization-layer-when
    # https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736
    def FreezeBatchNormLayers(self):
        
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm3d):
                module.eval()  
        return
    
    ###########################################################################
    # These functions are for displaying the kernels/filters of the convolutional
    # layers.
    
    # Extract all the kernels/filters of the convolutional layers of the VAE.
    # Output: - kernels
    def ExtractKernels(self):
        
        kernels = []
        
        # Looping over the different blocks of the CNN, for the convolutional
        # part of it. Each block is composed by: Conv2d, BatchNorm2d, LeakyReLU
        for block in self.encoder:
            # Looping over the three layers of each block.
            for layer in block:
                # If this is a CONVOLUTIONAL layer (and not a batchNorm or leakyRelu)
                if isinstance(layer, torch.nn.modules.Conv2d) or \
                   isinstance(layer, torch.nn.modules.Conv3d):
                    # Extract the kernels for current layer
                    current_kernels = layer.weight.detach().clone().cpu()
                    # Normalize
                    current_kernels_min = current_kernels.min()
                    current_kernels_max = current_kernels.max()
                    current_kernels     = (current_kernels - current_kernels_min)/(current_kernels_max- current_kernels_min)
                    # Save the kernel here
                    kernels.append(current_kernels)
                # Increment counter over number of layers.
        
        return kernels
    
    # Plot the kernels of the first convolutional layer
    def PlotFirstConvolutionKernels(self):
        
        kernels = self.ExtractKernels()
        
        fig, axarr = plt.subplots(1,1,figsize=(16, 16))
        img = make_grid(kernels[0])
        plt.imshow(img.permute(1, 2, 0))
        
        return kernels
    
    # Plot and print the kernels of the first convolutional layer
    def PlotAndPrintFirstConvolutionKernels(self, fileName):
        
        kernels = self.PlotFirstConvolutionKernels()
        plt.savefig(fileName + '.png') 
        
        return kernels
    
    ###########################################################################
    # These functions are for displaying the feature maps from the convolutional
    # layers.
    
    # Extract all the feature maps / activation maps for one image.
    # Input:  - x: image
    # Output: - activationMaps: a list containing all the feature maps, for
    #           each layer of the VAE
    def ExtractActivationMapsOfInput(self, x):
        
        # Where to put the activation maps
        activationMaps = []
        
        # Counter over the layers
        i = 0
        # Looping over the different blocks of the CNN, for the convolutional
        # part of it. Each block is composed by: Conv2d, BatchNorm2d, LeakyReLU
        for block in self.encoder:
            # Looping over the three layers of each block.
            for layer in block:
                # If this is the first layer of the first block, give as input
                # the image...
                if i == 0:
                    output_current_layer = layer(x)
                # ... otherwise give the output of the previous layer
                else:
                    output_current_layer = layer(output_current_layer)
                # If this is a CONVOLUTIONAL layer (and not a batchNorm or leakyRelu)
                if isinstance(layer, torch.nn.modules.Conv2d) or \
                   isinstance(layer, torch.nn.modules.Conv3d):
                    current_activationMap = output_current_layer.detach().clone().cpu()
                    current_activationMap_min = current_activationMap.min()
                    current_activationMap_max = current_activationMap.max()
                    current_activationMap     = \
                       (current_activationMap - current_activationMap_min)/(current_activationMap_max-current_activationMap_min)
                    # Save the activation map here
                    activationMaps.append(current_activationMap)
                # Increment counter over number of layers.
                i += 1
                
        return activationMaps
    
    # Plot one of the extracted feature maps / activation maps for an image.
    # Input:  - activationMaps: activation maps calculated with 'ExtractFeatureMapsOfInput'.
    #         - indexOfActivationMap: index of activation map to plot.
    @staticmethod
    def PlotSingleActivationMap(activationMaps, indexOfActivationMap):
        
        # Select the activation map of the desired block
        selectedActivationMap = activationMaps[indexOfActivationMap]
        
        # Defining the plot dimensions
        fig, axarr = plt.subplots(selectedActivationMap.shape[1] // 4,
                                  4,
                                  figsize=(16, selectedActivationMap.shape[1]))
        
        # Index over the subplot
        k=0
        
        # PLOTTING
        for i in range(selectedActivationMap.shape[1]//4):
            for j in range(4):
                axarr[i,j].imshow(selectedActivationMap[0,k,:,:].detach().cpu().numpy(), cmap='gray')
                k+=1  
                
        return
        
    # Does as PlotSingleFeatureMap, but also prints the map to file.
    # Input:  - activationMaps: activation maps calculated with 'ExtractFeatureMapsOfInput'.
    #         - indexOfActivationMap: index of activation map to plot.
    #         - fileName: name of file where to save the plot
    @staticmethod
    def PlotAndPrintSingleActivationMap(activationMaps, indexOfActivationMap, fileName):
        
        VAE.PlotSingleActivationMap(activationMaps, indexOfActivationMap)
        plt.savefig(fileName + '.png') 
        
        return
    
    # Plots all the the extracted feature maps / activation maps for an image.
    # Input:  - x: image
    def PlotAllActivationMaps(self, x):
        
        # Find the activation maps for the image
        activationMaps = self.ExtractActivationMapsOfInput(x)
        
        # Print them
        for i in range(len(activationMaps)):
            VAE.PlotSingleActivationMap(activationMaps, i)
            
        return activationMaps
    
    # Does as PlotAllActivationMaps, but also prints the maps to file.
    # Input:  - x: image
    def PlotAndPrintAllActivationMaps(self, x, baseFileName):
        
        # Find the activation maps for the image
        activationMaps = self.ExtractActivationMapsOfInput(x)
        
        # Print them
        for i in range(len(activationMaps)):
            VAE.PlotSingleActivationMap(activationMaps, i)
            plt.savefig(baseFileName + '_' + str(i) + '.png') 
            
        return activationMaps
    
    ###########################################################################
    # These functions are used to see what happens when a certain part of the
    # original image is covered.
    
    # Function to cover a part of the image.
    #@staticmethod
    #def CoverImagePart(x, windowSize = 3, )
    
    ###########################################################################
    # STATIC methods for loss calculation, printing of reconstructed images etc.
    
    @staticmethod
    def FindReconstructionLoss(reconstructedImages, realImages):
        # MSE calculated between actual images and reconstructed ones
        MSEReconstruction = F.mse_loss(reconstructedImages, realImages,size_average=False)#cuda()  
        return MSEReconstruction
    
    @staticmethod
    def FindKLDLoss(mu, logvar):        
        # Kullback-Leibler Divergence calculation, on mu 
        KLD               = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD
    
    # Function to find the losses of the VAE training
    # Inputs:
    # - mu and logvar: bottleneck of vae
    # - reconstructedImages: reconstructed images
    # - realImages: real images
    # Outputs:
    # - MSEReconstruction: reconstruction error
    # - KLD: Kullback-Leibler Divergence loss to force gaussianity
    @staticmethod
    def FindVAELosses(mu, logvar, reconstructedImages, realImages):
    
        # Kullback-Leibler Divergence calculation, on mu 
        KLD               = VAE.FindKLDLoss(mu, logvar)
        # MSE calculated between actual images and reconstructed ones
        MSEReconstruction = VAE.FindReconstructionLoss(reconstructedImages, realImages)    
        
        return KLD, MSEReconstruction
    
    # Print a group of real vs. reconstructed images
    # Inputs:
    # - realImages: original images
    # - reconstructedImages: reconstruction of the real images
    # Works for both grayscale and RGB images.
    # If images have 4 dimensions, we suppose the first one is of size = 1
    # and is a remaining from batch choice, so we concatenate on the
    # second dimension.
    @staticmethod
    def PrintRealVsReconstructedImages(realImages, reconstructedImages, outputFolderFile):
        
        if realImages.ndim == 2: 
            imagesToPrint = torch.cat([realImages, reconstructedImages])
        if realImages.ndim == 3:  
            imagesToPrint = torch.cat([realImages, reconstructedImages], axis = 1)
        elif realImages.ndim == 4:
            imagesToPrint = torch.cat([realImages, reconstructedImages], axis = 2)
            
        save_image(imagesToPrint, outputFolderFile)
        
        return
    
    ###########################################################################
    # SUMMARY FUNCTIONS to use in training/testing over epochs
    # These functions are for saving VAE-specific values to a 'SummaryHolder'
    # object.
    
    # This function brings from torch to numpy values VAE-specific values that
    # we want to save in the summary.
    @staticmethod
    def BringToNumpyVAEValuesForSummary(vaeLoss, ReconstructionLoss, KLDLoss, loss_tot, a_mu):
        
        # To numpy
        vaeLoss_numpy                    = vaeLoss.cpu().detach().numpy()
        ReconstructionLoss_numpy         = ReconstructionLoss.cpu().detach().numpy()
        KLDLoss_numpy                    = KLDLoss.cpu().detach().numpy()
        loss_tot_numpy                   = loss_tot.cpu().detach().numpy()
        a_mu_numpy                       = a_mu.cpu().detach().numpy()
        
        return vaeLoss_numpy, ReconstructionLoss_numpy, KLDLoss_numpy, loss_tot_numpy, a_mu_numpy
    
    # This function inserts VAE-specific values into the summary.
    @staticmethod
    def UpdateVAESummaries(summaryCurrentEpoch, vaeLoss_numpy, ReconstructionLoss_numpy,
                               KLDLoss_numpy, loss_tot_numpy, a_mu_numpy):
        # ADDING
        summaryCurrentEpoch.AppendValueInSummary('VAE_losses', vaeLoss_numpy)  
        summaryCurrentEpoch.AppendValueInSummary('Reconstruction_losses', ReconstructionLoss_numpy)  
        summaryCurrentEpoch.AppendValueInSummary('KLD_losses', KLDLoss_numpy)  
        summaryCurrentEpoch.AppendValueInSummary('Total_loss', loss_tot_numpy)  
        summaryCurrentEpoch.AppendValueInSummary('a_states', a_mu_numpy) 

        return summaryCurrentEpoch
    
    ###########################################################################
    # TRAINING AND TESTING FUNCTIONS
    
    # Perform forward pass and find losses
    def CallVAE(self, realImages):
        
        # Go through VAE's encoder and decoder
        reconstructedImages, z, mu, logvar = self.forward(realImages)
        # Find the losses on which to optimize
        KLDLoss, MSEReconstruction = VAE.FindVAELosses(mu, logvar, reconstructedImages, realImages)
        
        return reconstructedImages,z, mu, logvar, KLDLoss, MSEReconstruction
        
    
    # Function for training the VAE over a single batch
    def TrainVAEBatch(self, realImages, alpha = 1, learningRate = 0.001, weightDecay = 0.001, 
                 maxGradNorm = 300, optimizer = 'Adam'):
        
        # Perform forward pass and find losses
        reconstructedImages,z, mu, logvar, KLDLoss, MSEReconstruction = self.CallVAE(realImages)
        # Put together the two losses
        vaeLoss  = MSEReconstruction + alpha*KLDLoss
        
        # Select the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learningRate, weight_decay= weightDecay)
        
        # Perform optimization step
        optimizer.zero_grad()
        vaeLoss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), maxGradNorm)
        optimizer.step()
        
        return reconstructedImages, z, mu, logvar, KLDLoss, MSEReconstruction, vaeLoss
    
    @staticmethod
    def PrintRealVsReconstructedImageFromBatch(realImagesBatch, reconstructedImagesBatch, fileName, indexImageToPrintInBatch = 0):
        
        VAE.PrintRealVsReconstructedImages(realImagesBatch[indexImageToPrintInBatch, :, :, :], 
                                           reconstructedImagesBatch[indexImageToPrintInBatch, :, :, :], 
                                           fileName)
        
        return
    
    # Train the VAE over an epoch 
    def TrainVAEOverEpoch(self, trainingImages, 
                          outputFolder,
                          currentEpochNumber, batchSize,
                          decayRate, decaySteps, alpha = 1, learningRate = 0.001, 
                          weightDecay = 0.001, maxGradNorm = 300, optimizer = 'Adam'):
        
        # We are in TRAINING MODE
        self.train()
        # Number of data batches
        numberOfBatches = trainingImages.shape[0] // batchSize # takes floor value
        
        # Summary of training losses and values, for current epoch
        summaryTrainingCurrentEpoch = SH.SummaryHolder(summaryNamesCurrentEpochVAE)
        
        # Modify learning rate
        if currentEpochNumber > 0:
            globalStep   = currentEpochNumber
            learningRate = learningRate*np.power(decayRate, globalStep/decaySteps)
        # Put the current learning rate in the summary
        summaryTrainingCurrentEpoch.AppendValueInSummary('learning_rates', learningRate)    
            
        # Random image to print in this epoch
        randomBatchToPrint = np.random.randint(numberOfBatches)
        randomImageToPrintinBatch = np.random.randint(batchSize)
        
        #######################################################################
        # Looping over the batches of training data
        for i in range(numberOfBatches):
            
            print('Batch number ' + str(i+1) + ' out of ' + str(numberOfBatches))
            
            # Select slice corresponding to batch
            slc = slice(i * batchSize, (i + 1) * batchSize)
            
            # Select the batch for images
            # Original data dimension:
            # [batch_size, sequences_length, imageDimensionX, imageDimensionY]
            currentImagesBatch    = trainingImages[slc].to(device)
    
            # If there is no color channel, add one after batch size
            if len(currentImagesBatch.shape) == 3:
                currentImagesBatch = torch.unsqueeze(currentImagesBatch, 1)
                
            ################ CALL TO SINGLE BATCH TRAIN #######################
            reconstructedImagesBatch,a_seq, a_mu, a_var, KLDLoss, ReconstructionLoss, vaeLoss = \
               self.TrainVAEBatch(currentImagesBatch, alpha = alpha, learningRate = learningRate, 
                                     weightDecay = weightDecay)
            loss_tot = vaeLoss
                
            ###################################################################
            # Printing image reconstruction temporary results
            if i == randomBatchToPrint:
                VAE.PrintRealVsReconstructedImageFromBatch(currentImagesBatch, reconstructedImagesBatch, 
                                                       outputFolder + 'TRAIN_IMGS_' + str(currentEpochNumber) + '.png',
                                                       randomImageToPrintinBatch)
                
            ###################################################################
            # NUMPY VERSIONS
            vaeLoss_numpy, ReconstructionLoss_numpy, KLDLoss_numpy, loss_tot_numpy, a_mu_numpy = \
               VAE.BringToNumpyVAEValuesForSummary(vaeLoss, ReconstructionLoss, KLDLoss, loss_tot, a_mu)
            
            ###################################################################
            # TO SUMMARY
            summaryTrainingCurrentEpoch = VAE.UpdateVAESummaries(summaryTrainingCurrentEpoch, vaeLoss_numpy, 
                ReconstructionLoss_numpy, KLDLoss_numpy, loss_tot_numpy, a_mu_numpy)

            ###################################################################
            # Empty GPU from useless data
            del reconstructedImagesBatch
            del KLDLoss, ReconstructionLoss, vaeLoss
            del a_seq, a_mu, a_var
            del currentImagesBatch
            del loss_tot
            
            if device.type == "cuda":
                torch.cuda.empty_cache() 
            
            # End of batch loop
        #######################################################################  
        # Save the models
        torch.save(self.state_dict(), outputFolder + '/vae.torch')
        torch.save(self.state_dict(), outputFolder + '/vae_' + str(currentEpochNumber) + '.torch')
        
        return summaryTrainingCurrentEpoch, learningRate
    
    # Test the VAE over an epoch 
    def TestVAEOverEpoch(self, testingImages, outputFolder, currentEpochNumber, batchSize, alpha = 1):
        
        # We are in TESTING MODE
        self.eval()
        # Number of data batches
        numberOfBatches = testingImages.shape[0] // batchSize # takes floor value
        
        # Summary of testing losses and values, for current epoch
        summaryTestingCurrentEpoch = SH.SummaryHolder(summaryNamesCurrentEpochVAE)
            
        # Random image to print in this epoch
        randomBatchToPrint = np.random.randint(numberOfBatches)
        randomImageToPrintinBatch = np.random.randint(batchSize)
        
        #######################################################################
        # Looping over the batches of training data
        for i in range(numberOfBatches):
            
            print('Batch number ' + str(i+1) + ' out of ' + str(numberOfBatches))
            
            # Select slice corresponding to batch
            slc = slice(i * batchSize, (i + 1) * batchSize)
            
            # Select the batch for images
            # Original data dimension:
            # [batch_size, sequences_length, imageDimensionX, imageDimensionY]
            currentImagesBatch    = testingImages[slc].to(device)
    
            # If there is no color channel, add one after batch size
            if len(currentImagesBatch.shape) == 3:
                currentImagesBatch = torch.unsqueeze(currentImagesBatch, 1)
    
            ######################### CALL TO VAE #############################   
            reconstructedImagesBatch,a_seq, a_mu, a_var, KLDLoss, ReconstructionLoss = \
               self.CallVAE(currentImagesBatch)
                           
            vaeLoss  = ReconstructionLoss + alpha*KLDLoss
            loss_tot = vaeLoss
                
            ###################################################################
            # Printing image reconstruction temporary results
            if i == randomBatchToPrint:
                
                VAE.PrintRealVsReconstructedImageFromBatch(currentImagesBatch, reconstructedImagesBatch, 
                                                       outputFolder + '/TEST_IMGS_' + str(currentEpochNumber) + '.png',
                                                       randomImageToPrintinBatch)
    
            ###################################################################
            # NUMPY VERSIONS
            vaeLoss_numpy, ReconstructionLoss_numpy, KLDLoss_numpy, loss_tot_numpy, a_mu_numpy = \
               VAE.BringToNumpyVAEValuesForSummary(vaeLoss, ReconstructionLoss, KLDLoss, loss_tot, a_mu)
            
            ###################################################################
            # TO SUMMARY
            summaryTestingCurrentEpoch = VAE.UpdateVAESummaries(summaryTestingCurrentEpoch, vaeLoss_numpy, 
                ReconstructionLoss_numpy, KLDLoss_numpy, loss_tot_numpy, a_mu_numpy)
    
            # Empty GPU from useless data
            del reconstructedImagesBatch
            del KLDLoss, ReconstructionLoss, vaeLoss
            del a_seq, a_mu, a_var
            del currentImagesBatch
            del loss_tot
            
            if device.type == "cuda":
                torch.cuda.empty_cache() 
            
            # End of batch loop
            
        return summaryTestingCurrentEpoch
         
    # Train the VAE over a set of epochs 
    def TrainVAE(self, trainingImages, outputFolder, epochs, 
                 batchSize, decayRate, decaySteps, alpha = 1, learningRate = 0.001, 
                 weightDecay = 0.001, maxGradNorm = 300, optimizer = 'Adam'):
        
        # Summary of training, for all epochs
        summaryTrainingAllEpochs    = SHLAE.SummaryHolderLossesAcrossEpochs(summaryNamesAllEpochsVAE)
        
        # Loop over epochs and train
        for n in range(epochs):
            print('Epoch: ' + str(n))
            # Train over a single epoch
            summaryTrainingCurrentEpoch, learningRate = self.TrainVAEOverEpoch(trainingImages = trainingImages, 
                                                              outputFolder = outputFolder,
                                                              currentEpochNumber = n, 
                                                              batchSize = batchSize,
                                                              decayRate = decayRate, 
                                                              decaySteps = decaySteps, 
                                                              alpha = alpha, 
                                                              learningRate = learningRate, 
                                                              weightDecay = weightDecay, 
                                                              maxGradNorm = maxGradNorm, 
                                                              optimizer = optimizer)
            # Handle the losses over TRAINING epochs
            summaryTrainingAllEpochs.PerformFinalBatchOperations(summaryTrainingCurrentEpoch, outputFolder, filePrefix = 'TRAIN_')

        return summaryTrainingAllEpochs, learningRate
    
    # Train the VAE over a set of epochs, and perform testing on them too
    def TrainAndTestVAE(self, trainingImages, testingImages, outputFolder, epochs, 
                        batchSize, decayRate, decaySteps, alpha = 1, learningRate = 0.001, 
                        weightDecay = 0.001, maxGradNorm = 300, optimizer = 'Adam'):
        
        # Summary of training, for all epochs
        summaryTrainingAllEpochs    = SHLAE.SummaryHolderLossesAcrossEpochs(summaryNamesAllEpochsVAE)
        # Summary of testing, for all epochs
        summaryTestingAllEpochs     = SHLAE.SummaryHolderLossesAcrossEpochs(summaryNamesAllEpochsVAE)
        
        # Loop over epochs and train
        for n in range(epochs):
            print('Epoch: ' + str(n))
            print('TRAIN:')
            # Train over a single epoch
            summaryTrainingCurrentEpoch,  learningRate = self.TrainVAEOverEpoch(trainingImages = trainingImages, 
                                                              outputFolder = outputFolder,
                                                              currentEpochNumber = n, 
                                                              batchSize = batchSize,
                                                              decayRate = decayRate, 
                                                              decaySteps = decaySteps, 
                                                              alpha = alpha, 
                                                              learningRate = learningRate, 
                                                              weightDecay = weightDecay, 
                                                              maxGradNorm = maxGradNorm, 
                                                              optimizer = optimizer)
            # Handle the losses over TRAINING epochs
            summaryTrainingAllEpochs.PerformFinalBatchOperations(summaryTrainingCurrentEpoch, outputFolder, filePrefix = 'TRAIN_')
            
            print('TEST:')
            # Now test
            summaryTestingCurrentEpoch = self.TestVAEOverEpoch(testingImages = testingImages, 
                                                              outputFolder = outputFolder,
                                                              currentEpochNumber = n, 
                                                              batchSize = batchSize,
                                                              alpha = alpha)
            # Handle the losses over TESTING epochs
            summaryTestingAllEpochs.PerformFinalBatchOperations(summaryTestingCurrentEpoch, outputFolder, filePrefix = 'TEST_')
        
        return summaryTrainingAllEpochs, summaryTestingAllEpochs, summaryTrainingCurrentEpoch, summaryTestingCurrentEpoch, learningRate
    
    # Function to either load or to train and test the VAE and save it.
    # If the VAE was already trained, it is loaded from the path described in 
    # 'config' dictionary, otherwise it is trained from scratch.
    # INPUTS:
    # - config: configuration dictionary
    # - shuffledTrainingData: training data, shuffled
    # - shuffledTestingData: testing data, shuffled
    def PerformVAETrainingAndTesting(self, config, shuffledTrainingData, shuffledTestingData, outputFolderString = 'output_folder'):
        
        print('VAE training/testing beginning...')
    
        # ------------------ PERFORM THE VAE MAIN LOOP ON TRAIN DATA --------------    
        # Initial learning rate
        learningRate = config['lr_only_vae']
        
        if config['image_channels'] != 1:
            trainingImages = torch.swapaxes(shuffledTrainingData.images, 0, 1)
            testingImages  = torch.swapaxes(shuffledTestingData.images, 0, 1)
        else:
            trainingImages = shuffledTrainingData.images
            testingImages  = shuffledTestingData.images
        
        # Loading if VAE was already trained ...
        if config['VAE_already_trained'] == True:    
            # Search first for the file in the config[outputFolderString] folder, and if it
            # is not there, look in the parent directory
            vae_file_path = config[outputFolderString] + config['trained_VAE_file_name']
            if os.path.isfile(vae_file_path) == False:
                vae_file_path = os.path.dirname(os.path.dirname(vae_file_path))
                vae_file_path = vae_file_path + '/' + config['trained_VAE_file_name'] 
            print('File path of trained VAE:')
            print(vae_file_path)
            # load vae    
            self.load_state_dict(torch.load(vae_file_path))
            self.to(device)           
            # Summary names
            summaryNamesAllEpochsVAE = VAE.ReturnSummaryNamesAllEpochsVAE()
            # Initialize summary objects over training and testing
            summaryTrainingAllEpochs = SH.SummaryHolder(summaryNamesAllEpochsVAE)
            summaryTestingAllEpochs  = SH.SummaryHolder(summaryNamesAllEpochsVAE)
            
            summaryTrainingCurrentEpoch = self.TestVAEOverEpoch(testingImages = trainingImages, 
                                                              outputFolder = config[outputFolderString],
                                                              currentEpochNumber = config['only_vae_epochs']-1, 
                                                              batchSize = config['batch_size_VAE'],
                                                              alpha = config['alpha_VAEonly_training'])
            summaryTestingCurrentEpoch  = self.TestVAEOverEpoch(testingImages = testingImages, 
                                                              outputFolder = config[outputFolderString],
                                                              currentEpochNumber = config['only_vae_epochs']-1, 
                                                              batchSize = config['batch_size_VAE'],
                                                              alpha = config['alpha_VAEonly_training'])    
        # ... otherwise train the VAE.        
        else:  
            summaryTrainingAllEpochs, summaryTestingAllEpochs, summaryTrainingCurrentEpoch, \
                summaryTestingCurrentEpoch, learningRate = self.TrainAndTestVAE(
                trainingImages = trainingImages, testingImages = testingImages,
                outputFolder = config[outputFolderString],
                epochs = config['only_vae_epochs'], batchSize = config['batch_size_VAE'],
                decayRate = config['decay_rate'], decaySteps = config['decay_steps'], 
                alpha = config['alpha_VAEonly_training'], learningRate = learningRate, 
                weightDecay = config['weight_decay'], maxGradNorm = config['max_grad_norm_VAE'], 
                optimizer = 'Adam')
            
        self.FreezeBatchNormLayers()
        
        return summaryTrainingAllEpochs, summaryTestingAllEpochs, \
            summaryTrainingCurrentEpoch, summaryTestingCurrentEpoch  
    
    