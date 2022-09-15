
import torch
import json
import os

class ConfigurationHolder(object):
    """ This class a holder of the configuration for any type of file, also
        holding whether the GPU must be used or the CPU.
    """
    
    # Initialization of an empty dictionary
    def __init__(self):
        
        self.config = {}        
        return
        
    # Function to read the configuration from a python dictionary
    # Inputs:
    # - config: python dictionary containing the configuration
    def LoadConfigFromPythonDictionary(self, config):
        
        self.config = config
        return 
    
    # Function to read the configuration from a JSON file
    # Inputs:
    # - jsonFile: path to the JSON file where the configuration is stored
    def LoadConfigFromJSONFile(self, jsonFile):
        
        with open(jsonFile) as json_data_file:
            self.config = json.load(json_data_file)       
        return
        
    # Function to read the configuration from a txt file
    # Inputs:
    # fileName - Variable with the name of the file (full path) where the 
    #            dictionary is saved. The file must be a txt.
    def LoadConfigFromTxt(self, fileName):
        
        return
        
    # Function to read the configuration from a JSON file
    # Inputs:
    # fileName - Variable with the name of the file (full path) where the 
    #            dictionary is saved. The file must be JSON.
    def SaveConfigurationSettingsToJSONFile(self):
        
        # where to save
        path = self.config['output_folder'] + '/' + self.config['config_Filename'] + '.json'
        
        with open(path, "w") as outfile:
            json.dump(self.config, outfile, indent=2)        
        return
    
    # Save a txt file with the configuration settings
    # The name of the file should be in the configuration itself
    def SaveConfigurationSettingsToTxt(self):
        
        firstLine = True
        
        # where to save
        path = self.config['output_folder'] + '/' + self.config['config_Filename'] + '.txt'
    
        for key in self.config:
            
            # If it is the first key of config, cancel everything on file and write the 
            # first line on the file
            if firstLine == True:
                with open(path, 'w') as file:
                    line = key + ' : ' + str(self.config[key]) +'\n'
                    file.write(line)
            # Otherwise, just write afterward
            else:
                with open(path, 'a') as file:
                    line = key + ' : ' + str(self.config[key]) + '\n'
                    file.write(line)
                    
            firstLine = False
        
        return
    
    # Create a folder for the output, reading the 'output_folder' variable in
    # the configuration.
    def MakeFolderForOutput(self):
        
        # Create folder for output, if it does not exist
        if not os.path.exists(self.config['output_folder']):
            os.makedirs(self.config['output_folder'])
        
        return
    
    # Redefines the output folder
    def RedefineOutputFolder(self, output_folder):
        
        self.config['output_folder'] = output_folder
        
        return
    
    # Redefines and creates the output folder
    def RedefineAndCreatesOutputFolder(self, output_folder):
        
        self.RedefineOutputFolder(output_folder)
        self.MakeFolderForOutput()
        
        return
    
    # Redefine the output folder by adding the path to the base folder in
    # front of it.
    # INPUTS:
    # - baseFolderPath: path to base folder
    # Example:
    # self.config['output_folder'] = "/output/"
    # baseFolderPath = "C:/user/base/"
    # -> return "C:/user/base/output/"
    def AddBasePathToOutputFolder(self, baseFolderPath):
        
        self.RedefinePathAddingBaseFolder('output_folder', baseFolderPath)
        
        return
    
    # Redefines the output folder and creates it
    def AddBasePathAndCreateFolderForOutput(self, baseFolderPath):
        
        # Redefine
        self.AddBasePathToOutputFolder(baseFolderPath)
        # Create folder for outputs / check if it exists
        self.MakeFolderForOutput()    
        
        return
    
    # Define if you want to force cpu usage
    # Input: 
    # - force_cpu_use: variable defining if we want to force the CPU usage
    def ConfigureGPUSettings(self, force_cpu_use = False):
        
        self.config['force_cpu_use'] = force_cpu_use;
                
        return 
    
    # Function to retrieve if GPU or CPU to use.
    def GetDeviceVariable(self):
        
        # GPU or CPU?
        # If the key eists in the dictionary and it is set to True
        if 'force_cpu_use' in self.config.keys() and self.config['force_cpu_use'] == True:
            # This is to force CPU or GPU usage
            device = torch.device("cpu")
        else:
            # This automatically chooses the device that is available:
            # - if there is a GPU, it considers it
            # - otherwise, it takes the CPU
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        return device
    
    # Redefine a path adding a prefix path to it.
    # INPUTS:
    # - nameOfConfigurationVariable: name of the variable containing the path
    # - baseFolderPath: path to base folder
    def RedefinePathAddingBaseFolder(self, nameOfConfigurationVariable, baseFolderPath):
        
        self.config[nameOfConfigurationVariable] = baseFolderPath + self.config[nameOfConfigurationVariable]
        
        return
    
    # This is a static method that can be used to create a configuration object
    # from a json file, create a folder for its output and save the configuration 
    # with the name defined in the json file itself.
    # INPUTS:
    # - jsonConfigurationFile: path to the json configuration file.
    # OUTPUTS:
    # - configHolder: created configuration object.
    @staticmethod
    def PrepareConfigHolder(jsonConfigurationFile):

        configHolder = ConfigurationHolder.PrepareConfigHolderWithoutOutputFolder(jsonConfigurationFile)
        # Create folder for outputs / check if it exists
        configHolder.MakeFolderForOutput()    
        # Save to file the configurations
        configHolder.SaveConfigurationSettingsToTxt()

        return configHolder
    
    # Similar to 'PrepareConfigHolder', but we suppose the configuration file does not
    # contain an output folder.
    @staticmethod
    def PrepareConfigHolderWithoutOutputFolder(jsonConfigurationFile):
        
        # This part of the code configures the settings of the models (e.g., how the 
        # VAE is, how big the latent states are, where to save etc.).
        # A dictionary named "config" is created, containing all the configuration 
        # parameters.
        # Go to the configuration folder to manually define these settings. 
        
        # Initialize configuration holder
        configHolder        = ConfigurationHolder() 
        # Fill configHolder with the configuration from a json file
        configHolder.LoadConfigFromJSONFile(jsonConfigurationFile)  

        return configHolder
    
    # Similar to 'PrepareConfigHolder', but we suppose the json file does not contain
    # the full path, which has to be built
    @staticmethod
    def PrepareConfigHolderWithOutputFolderToAddBase(jsonConfigurationFile, baseFolderPath):

        # This part of the code configures the settings of the models (e.g., how the 
        # VAE is, how big the latent states are, where to save etc.).
        # A dictionary named "config" is created, containing all the configuration 
        # parameters.
        # Go to the configuration folder to manually define these settings. 

        configHolder = ConfigurationHolder.PrepareConfigHolderWithoutOutputFolder(jsonConfigurationFile)
        configHolder.AddBasePathAndCreateFolderForOutput(baseFolderPath)
        # Save to file the configurations
        configHolder.SaveConfigurationSettingsToTxt()

        return configHolder