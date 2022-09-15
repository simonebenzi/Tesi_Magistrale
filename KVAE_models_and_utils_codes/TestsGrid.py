# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:30:47 2021

@author: giulia.slavic
"""

import numpy as np
import os
import itertools
import scipy.io as sio

# A Class for defining a grid of testing parameters.
# This grid can be created and used to perform multiple-paramters testing
# on a variety of models.

class TestsGrid(object):
    
    # Initialization of the tests grid.
    def __init__(self, parameterValues, parameterNames):
        
        # To create the grid with the values
        self.CreateParametersGrid(parameterValues)
        # Just the names of the parameters in the grid
        self.parameterNames     = parameterNames
        # Number of tests
        self.numberOfTests      = self.grid.shape[0]
        self.numberOfParameters = self.grid.shape[1]
        
        return
    
    # Initialization of the tests grid from a configHolder object.
    # INPUTS:
    # - configMultipleParams: configHolder object with the parameters.
    #      Values in the configHolder must be saved as strings separated
    #      by a comma:
    #      e.g.: '1,2,4,20'
    # OUTPUTS:
    # - testsGrid: the created grid
    @staticmethod
    def InitializeTestsGridFromConfigHolder(configMultipleParams):
        
        # Where to put the values of the parameters
        parameterValues = []
        # Where to put the names of the parameters
        parameterNames  = []
        # Looping of the params
        for parameterName in configMultipleParams:
            # Save only if this is not an element of the configuration file used
            # to set the name of the .txt file to produce and of the folder where
            # to produce it (i.e., not parameter values).
            if parameterName != 'config_Filename' and parameterName != 'output_folder':
                # Insert names
                parameterNames.append(parameterName)
                # Insert values
                parameterValue = configMultipleParams[parameterName]
                # Separate the values, if they are in a string
                if isinstance(parameterValue, str):
                    parameterValue = [float(f) for f in parameterValue.split(',')]
                # Take it as it is, if it is an array
                elif  type(parameterValue) is np.ndarray:
                    parameterValue = parameterValue
                # Insert values
                parameterValues.append(parameterValue)
            
        testsGrid = TestsGrid(parameterValues, parameterNames)
        
        return testsGrid
    
    # Function to create a grid of parameters from a list of values for each parameter.
    # INPUTS:
    # - parameterValues: a list containing in each element the values to consider 
    #                  for a particular parameter.
    #                  Type: list of 1D arrays.
    # OUTPUTS:
    # - parametersGrid: a grid containing the values of the different parameters to 
    #                 set. The number of rows corresponds to the number of 
    #                 different attempts one wants to perform; the number of 
    #                 columns to the number of parameters.
    #                 Type: numpy 2D array.
    # HOW IT WORKS:
    # 1) Initializing the empty grid after counting how many attempts and how many
    #    parameters it should contain. Number of attempts is a consequence of number
    #    of parameters and their dimension.
    # 2) Initializing the variables to toggle between the different parameter values.
    #    There should be as many toggle variables as the number of parameters and 
    #    they are set to toggle in succession between the different values that 
    #    parameter can take (parameterValues[i]).
    #    Since a 'itertools' variable does not have a function to retrieve its 
    #    current state, the current state is initialized in 'currentValues'. It is
    #    initialized with zeros, since it will be later set in the first loop
    #    iteration of phase 4).
    # 3) Setting the remainders that will tell in phase 4) when to toggle between
    #    parameters.
    #    Remember that if we have for example two parameters that can assume 2 and 3
    #    values respectively, we want to have them change like this:
    #    P1  P2
    #    1   1
    #    2   1
    #    3   1
    #    1   2
    #    2   2
    #    3   2
    #    So the reminder for toggling parameter i should be equivalent to the 
    #    multiplication of the number of values that can be assumed by the 
    #    parameters 1 ... i-1
    # 4) Inserting values in the grid, toggling using the defined toggle variables
    #    ('toggles') and the calculated reminders ('valuesForRemainders') 
    #    (i.e., we toggle to the next value of a parameter, when the reminder
    #    associated to it is zero).
    def CreateParametersGrid(self, parameterValues):
        
        # 1) INITIALIZATION GRID
        # Number of parameters
        numberOfParameters       = len(parameterValues)
        # Number of total experiments to perform
        numberOfTotalExperiments = 1
        for i in range(numberOfParameters):
            numberOfTotalExperiments *= len(parameterValues[i])
            
        # Creating the grid of parameters: as many rows as the total number of 
        # experiments and as many columns as the number of parameters
        parametersGrid = np.zeros((numberOfTotalExperiments, numberOfParameters))
        
        # 2) INITIALIZATION of TOGGLE VARIABLES
        # Create toggle variables
        toggles = []
        for i in range(numberOfParameters):
            currentToggle = itertools.cycle(parameterValues[i])
            toggles.append(currentToggle)
            
        # Where to put current value over iterator
        currentValues = np.zeros(numberOfParameters)
        
        # 3) REMAINDERS SETTER
        # Value to calculate remainder
        valuesForRemainders   = []
        for i in range(numberOfParameters):
            if i == 0:
                currentValueForReminders = 1
            else:
                currentValueForReminders *= len(parameterValues[i-1])
            valuesForRemainders.append(currentValueForReminders)
            
        # 4) INSERTING VALUES IN THE GRID
        # Inserting the values in the grid
        # Looping over the rows (number of experiments)
        for i in range(numberOfTotalExperiments):
            # Looping over the columns (number of parameters)
            for j in range(numberOfParameters):
                # Toggling?
                shouldIToggle             = np.remainder(i,valuesForRemainders[j])
                # If I should toggle (i.e, shouldIToggle = 0)
                if shouldIToggle == 0:
                    currentValues[j] = next(toggles[j])
                # Insert parameter value
                parametersGrid[i,j]       = currentValues[j]
                
        # Save the grid in the object
        self.grid = parametersGrid
                
        return  
    
    # Create a child folder of a base output folder, based on a conidered
    # row of the grid, i.e., giving to the folder the name of the parameters
    # and their corresponding value for that row of the grid.
    # INPUTS:
    # - baseOutputFolder: base folder where the child folder will be nested in
    # - gridRow: which row of the tests grid to consider
    # OUTPUTS:
    # - newOutputFolder: child folder nested in the base folder, with name
    #   the parameters in the grid and the respective values assumed in the row.
    #   e.g., '/1_learningRate_0.5_gradientCutting_2/'
    def DefineOutputFolderBasedOnRowOfGrid(self, baseOutputFolder, gridRow):
        
        newOutputFolder = baseOutputFolder
        
        # Setting the current parameters
        for gridColumn in range(self.numberOfParameters):
            # Current parameter name and value
            currentParameterName             = self.parameterNames[gridColumn]
            currentParameterValue            = self.grid[gridRow,gridColumn]
            # Change the name of the output folder so that it adds the defined parameters
            if gridColumn == 0:
                # At the beginning, add the row number in the grid
                newOutputFolder  = baseOutputFolder + '/' + str(gridRow).zfill(4)
            # Add name of parameters and their value in the row
            newOutputFolder      = newOutputFolder  + '_' + currentParameterName + '_' + str(currentParameterValue) + '_'
            # At the end, add a slash
            if gridColumn == self.numberOfParameters-1:
                newOutputFolder  = newOutputFolder + '/' 
                                               
        # Create folder if it did not exist
        if not os.path.exists(newOutputFolder):
             os.makedirs(newOutputFolder)
        
        return newOutputFolder
    
    # A function to assign to a dictionary the values of a row of the grid, 
    # with the corresponding names.
    # INPUTS:
    # - dictionary: could be a configuration file to modify on the fly;
    # - gridRow: which row of the tests grid to consider.
    # OUTPUTS:
    # - newDictionary: modified dictionary
    def AssignToDictionaryTheValuesOfGridRow(self, dictionary, gridRow):
        
        newDictionary = dictionary.copy() # remember the .copy() or will just take pointer
        
        # Setting the current parameters
        for gridColumn in range(self.numberOfParameters):
            
            # Current parameter name and value
            currentParameterName             = self.parameterNames[gridColumn]
            currentParameterValue            = self.grid[gridRow,gridColumn]
            # Taking the parameter from the grid
            newDictionary[currentParameterName] = currentParameterValue

        return newDictionary
    
    # A function to save the grid values to MATLAB.
    # INPUTS:
    # - outputFolder: folder where to put the saving of the grid.
    # - fileName: name of the grid.
    def SaveGridValuesToMATLAB(self, outputFolder, fileName):
        
        sio.savemat(outputFolder + fileName + '.mat', {fileName: self.grid})
        
        return
    
    # A function to save the grid names to MATLAB.
    # INPUTS:
    # - outputFolder: folder where to put the saving of the grid.
    # - fileName: name of the grid.
    def SaveGridNamesToMATLAB(self, outputFolder, fileName):
        
        sio.savemat(outputFolder + fileName + '_names' + '.mat', {fileName + '_names': self.parameterNames})
        
        return
    
    # A function to save the grid of parameters to MATLAB.
    # This saves both the grid values and the names of the parameters.
    # INPUTS:
    # - outputFolder: folder where to put the saving of the grid.
    # - fileName: name of the grid.
    def SaveGridToMATLAB(self, outputFolder, fileName):
        
        self.SaveGridValuesToMATLAB(outputFolder, fileName)
        self.SaveGridNamesToMATLAB(outputFolder, fileName)
        
        return