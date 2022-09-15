
# This is a Summary Holder object that is specific for losses across epochs, 
# i.e., storing values that are the average of one epoch loss, considered
# across different epochs.

###############################################################################

import numpy as np
import scipy.io as sio

from KVAE_models_and_utils_codes import PlotGraphs_utils          as PG
from KVAE_models_and_utils_codes import SummaryHolder             as SH

###############################################################################

class SummaryHolderLossesAcrossEpochs(SH.SummaryHolder):
    """ This class holds the summary data from a training phase, and updates it.
    """
    
    # Initialization of the summary data.
    def __init__(self, summaryNames):
        
        super(self.__class__, self).__init__(summaryNames)
        
        return
    
    # To use when, arrived at the end of an epoch, the loss of another
    def AppendToOverallSummaryMeanValuesOfCurrentEpochSummary(self, currentEpochSummary):
        
        # All the keys of the overall summary
        key_list = list(self.summary.keys())
        
        # Looping over all the keys of the overall summary
        for i in range(len(self.summary)):
            # Current key name
            current_key = key_list[i]
            # If the key is also in the current epoch summary
            currentKeyIsInCurrentEpochSummary       = current_key in currentEpochSummary.summary.keys()
            # If the corresponding value is not empty in the epoch summary
            currentKeyValueInEpochSummaryIsNotEmpty = len(currentEpochSummary.summary[current_key]) != 0 
            # If previous statements are true ...
            if currentKeyIsInCurrentEpochSummary and currentKeyValueInEpochSummaryIsNotEmpty:
                # ... average the values for the current epoch
                currentEpochAverage = np.mean(currentEpochSummary.summary[current_key], axis=0)
                self.AppendValueInSummary(current_key, currentEpochAverage)
            
        return

    # Plot one dimensional elements in the summary over their temporal values
    # This is good to plot losses, etc
    def PlotValuesInSummaryAcrossTime(self, outputFolder, filePrefix = ''):
        
        # All the keys of the overall summary
        key_list = list(self.summary.keys())
        
        # Looping over the elements in the dictionary
        for i in range(len(self.summary)):
            # Name of folder + file
            path_name = outputFolder + '/' + filePrefix + key_list[i] + '.png'
            # Plot the loss over all epochs
            PG.plot_loss(loss = self.summary[key_list[i]], file = path_name, title = key_list[i])
        
        return
    
    # Save the values to matlab
    def BringValuesToMatlab(self, outputFolder, filePrefix = ''):
        
        # All the keys of the overall summary
        key_list = list(self.summary.keys())
        
        # Looping over the elements in the dictionary
        for i in range(len(self.summary)):
            # Current key
            currentKey = key_list[i]
            # Values
            currentKeyValues = self.summary[currentKey]
            # Perform the reshaping of data
            # Name of folder + file
            path_name  = outputFolder + '/' + filePrefix + currentKey + '.mat'
            # Plot the loss over all epochs
            sio.savemat(path_name, {key_list[i]: currentKeyValues})
        
        return
    
    def PerformFinalBatchOperations(self, currentEpochSummary, outputFolder, filePrefix = ''):
        
        # Add the mean of the losses of the current epoch to the overall summary
        self.AppendToOverallSummaryMeanValuesOfCurrentEpochSummary(currentEpochSummary)
        # Plot losses
        self.PlotValuesInSummaryAcrossTime(outputFolder, filePrefix)
        # Save losses to matlab
        self.BringValuesToMatlab(outputFolder, filePrefix)
        
        return 