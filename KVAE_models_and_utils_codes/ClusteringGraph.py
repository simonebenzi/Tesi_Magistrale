
# This script contains functions for the ClusteringGraph CLASS
# This CLASS can be used to train a GNG (still to be added), 
# obtain a clustering graph, or read a clustering graph.

import numpy as np
import mat4py

from scipy.io import savemat

import torch
import copy 
import unittest

from KVAE_models_and_utils_codes import Distance_utils   as d_utils
from ConfigurationFiles          import Config_GPU as ConfigGPU

###############################################################################
# ---------------------------- GPU setting  ----------------------------------
###############################################################################

# GPU or CPU?
configGPU = ConfigGPU.ConfigureGPUSettings()
device    = ConfigGPU.DefineDeviceVariable(configGPU)

###############################################################################
# ---------------------------  ClusteringGraph   ------------------------------
###############################################################################

class ClusteringGraph(object):
    
    # Attributes of the class:
    # - num_clusters
    # - clustersSequence
    # - nodesMean
    # - nodesCov
    # - nodesCovPred (optional)
    # - nodesCovD (optional)
    # - additionalClusterInfo (optional)
    # - nodesDistanceMatrix
    # - nodesVicinityMatrix
    # - datanodes
    # Transition matrices:
    # - transitionMat
    # - windowedtransMatsTime
    # - transMatsTime
    # - maxClustersTime
    # For GNG training (still to put):
    # - E
    # - utility
    # - C
    # - t
    
    ###########################################################################
    # FUNCTIONS TO LOAD AN ALREADY TRAINED GRAPH FROM MATLAB
    
    # Initialize the clustering by reading it from a MATLAB file.
    # INPUTS:
    # - clusteringFile: path to the MATLAB file where the clustering is saved.
    # - nodesMeanName: name with which the nodesMean field is stored.
    # - nodesCovName: name with which the nodesCov field is stored.
    def LoadGraphFromMATLAB(self, clusteringFile, nodesMeanName = 'nodesMean', nodesCovName = 'nodesCov'):
        
        # Load the file from MATLAB
        clustering_data = mat4py.loadmat(clusteringFile)
            
        print('Extracting clustering features')
        # Extract the features related to clustering
        self.Extract_clustering_data_features(clustering_data, nodesMeanName, nodesCovName)
        
        return
    
    # Function to load an already trained graph using a configuration file 
    # (which needs to contain the state dimension) and from the path to the
    # clustering file.
    # INPUTS:
    # - config: configuration file, that must contain the 'dim_a' field, with
    #           the dimension of the KVAE 'a' latent state, and the 
    #           'deleteThreshClusterSmoothing' field, which defines below which
    #           threshold of consecutive occurrencies a cluster assignment can 
    #           be smoothed.
    # - clusteringFile: path to the MATLAB file where the clustering is saved.
    # - nodesMeanName: name with which the nodesMean field is stored.
    # - nodesCovName: name with which the nodesCov field is stored.
    # OUTPUTS:
    # - clusterGraph: the clustering.
    # - vectorOfKeptClusters: which clusters of the original clustering have 
    #   been kept after the smoothing operation.
    @staticmethod
    def PrepareClusterGraphFromConfigFileAndClusteringFileWithSmoothing(config, clusteringFile, 
                                                                        nodesMeanName = 'nodesMean', nodesCovName = 'nodesCov'):
        
        # Create empty graph
        clusterGraph = ClusteringGraph(config['dim_a'])
        print('Loading graph from MATLAB')
        # Load clustering from file
        clusterGraph.LoadGraphFromMATLAB(clusteringFile = clusteringFile, 
                                         nodesMeanName = nodesMeanName, nodesCovName = nodesCovName)
        print('Smoothing the clustering')
        # Smoothen the clustering assignment
        vectorOfKeptClusters = clusterGraph.SmoothClusterAssignmentsAndRecalculateProperties(
            deleteThreshold = config['deleteThreshClusterSmoothing'])
        
        # Print all information related to the loaded cluster
        clusterGraph.print()
        
        return clusterGraph, vectorOfKeptClusters
    
    def PrepareClusterGraphFromConfigFileAndClusteringFile(config, clusteringFile, 
                                                           nodesMeanName = 'nodesMean', nodesCovName = 'nodesCov'):
        
        # Create empty graph
        clusterGraph = ClusteringGraph(config['dim_a'])
        print('Loading graph from MATLAB')
        # Load clustering from file
        clusterGraph.LoadGraphFromMATLAB(clusteringFile = clusteringFile, 
                                         nodesMeanName = nodesMeanName, nodesCovName = nodesCovName)
        
        print('No smoothing')
        vectorOfKeptClusters = np.ones(clusterGraph.num_clusters)
        
        # Print all information related to the loaded cluster
        clusterGraph.print()
        
        return clusterGraph, vectorOfKeptClusters
    
    # Function to load an already trained graph using a configuration file 
    # (which needs to contain the state dimension and the path to the 
    # configuration file).
    # INPUTS:
    # - config: configuration file, that must contain the 'dim_a' field, with
    #           the dimension of the KVAE 'a' latent state, and the 
    #           'deleteThreshClusterSmoothing' field, which defines below which
    #           threshold of consecutive occurrencies a cluster assignment can 
    #           be smoothed. Additionally, it must contain a field
    #           'clustering_data_file', with the path to the clustering
    #           saved in MATLAB.
    # - nodesMeanName: name with which the nodesMean field is stored.
    # - nodesCovName: name with which the nodesCov field is stored.
    # OUTPUTS:
    # - clusterGraph: the clustering.
    # - vectorOfKeptClusters: which clusters of the original clustering have 
    #   been kept after the smoothing operation.
    @staticmethod
    def PrepareClusterGraphFromConfigFile(config, nodesMeanName = 'nodesMean', nodesCovName = 'nodesCov'):
        
        clusterGraph, vectorOfKeptClusters = ClusteringGraph.PrepareClusterGraphFromConfigFileAndClusteringFileWithSmoothing(
                                                           config = config, clusteringFile = config["clustering_data_file"], 
                                                           nodesMeanName = nodesMeanName, nodesCovName = nodesCovName)        
        return clusterGraph, vectorOfKeptClusters
    
    ###########################################################################
    # FUNCTIONS FOR EXTRACTING FEATURES FROM A LOADED CLUSTERING
    
    # Function to bring nodes covariances from a list to an array.
    # INPUTS: 
    # - nodesCovList: nodes covariances as list.
    # OUTPUTS:
    # - nodesCov: nodes covariances as array
    def Bring_nodesCov_from_list_to_array(self, nodesCovList):
        
        nodesCov = []
        for i in range(self.num_clusters):
            currCov  = np.array(nodesCovList[i])
            nodesCov.append(currCov)            
        return nodesCov

    # Extracting the features related to training clustering
    # INPUTS:  
    # - clustering_data: dictionary loaded from a MATLAB file.
    # - nodesMeanName: name with which the nodesMean field is stored.
    # - nodesCovName: name with which the nodesCov field is stored.
    # OUTPUTS: 
    # - (self) (with features extracted)
    # Extracted features: see the list at the beginning of the class.
    def Extract_clustering_data_features(self, clustering_data, nodesMeanName = 'nodesMean', nodesCovName = 'nodesCov'):
        
        # Extracting the number of clusters, mean of clusters, and synchronized data
        self.num_clusters     = np.array(clustering_data['net']['N'])
        self.nodesMean        = np.array(clustering_data['net'][nodesMeanName])
        #self.data_m          = np.array(clustering_data['net']['data_sync'])
        clustersSequence      = np.array(clustering_data['net']['dataColorNode'])
        
        # shift cluster sequence so that first cluster is always 0
        # (as MATLAB starts from 1 instead)
        firstClusterIndex     = np.min(clustersSequence)
        clustersSequence      = clustersSequence - firstClusterIndex
        self.clustersSequence = np.squeeze(clustersSequence)
    
        # Extracting the cluster covariances ...
        nodesCovTemp    = clustering_data['net'][nodesCovName] 
        # ... and putting them in a numpy array instead of a list           
        self.nodesCov   = self.Bring_nodesCov_from_list_to_array(nodesCovTemp)
        # From list to array
        self.nodesCov   = np.stack(self.nodesCov)
        
        # If there are additional info in the cluster
        if 'additionalClusterInfo' in clustering_data['net'].keys():
            self.additionalClusterInfo = np.array(clustering_data['net']['additionalClusterInfo'])
            
        #######################################################################
        # Additional features to the base ones     
        if 'nodesCovPred' in clustering_data['net'].keys():
            nodesCovPredTemp    = clustering_data['net']['nodesCovPred'] 
            self.nodesCovPred   = self.Bring_nodesCov_from_list_to_array(nodesCovPredTemp)
        else:
            self.nodesCovPred   = None
        if 'nodesCovD' in clustering_data['net'].keys():
            nodesCovDTemp       = clustering_data['net']['nodesCovD'] 
            self.nodesCovD      = self.Bring_nodesCov_from_list_to_array(nodesCovDTemp)
        else:
            self.nodesCovD      = None
            
        if 'startingPoints' in clustering_data['net'].keys():
            startingPoints      = clustering_data['net']['startingPoints']
            # If starting points are calculated with MATLAB, they might be shifted
            # of one value forwards, so we must change this.      
            print('Starting points:')
            print(startingPoints)
            if hasattr(startingPoints, "__len__"):
                print('starting point is list')
                startingPoints = np.array(startingPoints)
                startingPoints = [x-1 for x in startingPoints]
                self.startingPoints = np.array(startingPoints)
            else:
                print('starting point is NOT list')
                self.startingPoints = []
                self.startingPoints.append(0)           
        else: # if no field, set it as a single value = 0 (i.e., suppose single trajectory)
            self.startingPoints = []
            self.startingPoints.append(0)
                 
        #######################################################################
        # Transition matrices     
        self.CalculateAllTransitionMatrices()    
            
        if 'vicinityTransitionMat' in clustering_data['net'].keys():
            vicinityTransitionMat = np.array(clustering_data['net']['vicinityTransitionMat'])
            self.vicinityTransitionMat = vicinityTransitionMat
            self.CalculateTransitionMatrixExtendedThroughVicinityMatrix()
        else:
            self.vicinityTransitionMat = None
        
        #######################################################################
        # Transition matrices     
        self.CalculateAllTransitionMatrices()
        #self.FindNodesDistanceMatrix()
        #self.FindNodesVicinityMatrix()

        self.FindNodesBhattacharyaDistanceMatrix()
        
        return
    
    # Function to extract mean value and standard deviation value of training
    # data from the MATLAB file containing the trained clustering.
    # INPUTS:
    # - clusteringFile: path to the MATLAB file where the clustering is saved.
    def Extract_mean_std_of_data(self, clusteringFile):
        
        clustering_data = mat4py.loadmat(clusteringFile)
        
        self.X_mean = np.array(clustering_data['net']['X_mean'])
        self.X_std  = np.array(clustering_data['net']['X_std'])
        
        return
    
    ###########################################################################
    # FUNCTIONS to find the beginning or ending time of a trajectory inside
    # the 'self.clustersSequence' field, given the time instant in it.
    
    # Find the time instant of beginning of the current trajectory.
    # INPUTS:
    # - currentTimeInstant: current time instant considering having all the 
    #                       trajectories stacked along one dimension.
    # OUTPUTS:
    # - currentTrajBeginning: time instant of beginning of the current trajectory.
    def FindCurrentTrajectoryBeginningTime(self, currentTimeInstant):
        
        # If there is a single trajectory, the starting point can only be = 0
        if len(self.startingPoints) == 1:
            currentTrajBeginning = 0
        # Otherwise look for those starting points lower than the current time instant
        # and pick the maximum of them.
        else: 
            currentTrajBeginning = self.startingPoints[self.startingPoints < currentTimeInstant].max()
        
        return currentTrajBeginning
    
    # Find the time instant of ending of the current trajectory.
    # INPUTS:
    # - currentTimeInstant: current time instant considering having all the 
    #                       trajectories stacked along one dimension.
    # OUTPUTS:
    # - currentTrajEnding: time instant of ending of the current trajectory.
    def FindCurrentTrajectoryEndingTime(self, currentTimeInstant):
        
        # If there is a single trajectory, take the final length
        # of the cluster assignments element
        if len(self.startingPoints) == 1:
            currentTrajEnd = len(self.clustersSequence)
            return currentTrajEnd
        else:
            # Find the trajectory beginnings after the current time instant
            allNextTrajBeginning = self.startingPoints[self.startingPoints > currentTimeInstant]
        
        # If this is the last trajectory and so there are no consequent starting points,
        # (so the array 'nextTrajBeginning' is empty), take the final length
        # of the cluster assignments element
        if not allNextTrajBeginning.size:
            currentTrajEnd    = len(self.clustersSequence)
        # Otherwise find the min and add -1 to the next trajectory beginning
        else:
            # Find next trajectory beginning time
            nextTrajBeginning = allNextTrajBeginning.min()
            # add -1 for current trajectory ending
            currentTrajEnd    = nextTrajBeginning - 1
        
        return currentTrajEnd
    
    ###########################################################################
    # FUNCTIONS FOR CALCULATING THE TRANSITION MATRICES FROM THE FIELD
    # 'self.clustersSequence'.
    
    # Function that calls all the subfunctions for calculating all the transition
    # matrices and related fields.
    def CalculateAllTransitionMatrices(self):
        
        # Find transition matrix
        self.FindTransitionMatrix()
        
        # Find temporal transition matrices
        self.CalculateTemporalTransitionMatrices()
        # Find max time spent in each cluster
        self.CalculateMaxClustersTime()
        # Modify the temporal transition matrices so that they are smoothed
        # over a window of time (less overfitting).
        self.CalculateWindowedTemporalTransitionMatrices()
        
        return
    
    # Function to calculate the transition matrix from the 'self.clustersSequence'
    # field.
    def FindTransitionMatrix(self):
        
        # Empty transition matrix
        transitionMat = np.zeros((self.num_clusters,self.num_clusters));
        
        #Total length of training data
        trainingDataLength = len(self.clustersSequence)
        
        # Count transitions from one cluster to another one
        for t in range(trainingDataLength - 1):
            # Count the transition, if we are not at the last time instant
            # of a trajectory.
            if t + 1 not in self.startingPoints:
                transitionMat[self.clustersSequence[t],self.clustersSequence[t+1]] += 1;
            
        # Normalize transition matrix
        for i in range(self.num_clusters):
            transitionMat[i,:] = transitionMat[i,:]/np.sum(transitionMat[i,:])
        
        self.transitionMat = transitionMat;
        
        return
    
    # Input and Output: net structure obtained with GNG clustering
    # The function creates a set of transition matrices, one for each time value
    # from t to tMax, being tMax the time we have already spent in a node. 
    def CalculateTemporalTransitionMatrices(self):
    
        ## Temporal Transition Matrix
        
        # Total length of training data
        currLength    = self.clustersSequence.shape[0]
        
        # Find max number of time instants before a zone change 
        tMax          = self.FindMaxTimeBeforeTransition()
        
        # Find TIME TRANSITION MATRICES for follower
        self.FindTimeTransitionMatrices(tMax, currLength)
        
        return
    
    # Function to find the max overall time spent in a cluster.
    def FindMaxTimeBeforeTransition(self):

        # Initialise the max to 1
        tMax = 1
        
        arrayStartingPoints = np.asarray(self.startingPoints)
            
        # Looping over the time instants of the trajectory
        timeInstantsCurrentTrajectory = self.clustersSequence.shape[0]
        for t in range(timeInstantsCurrentTrajectory):
        
            # Zone at current time
            newZone = self.clustersSequence[t]
            
            if t in arrayStartingPoints:
                # Initialise the length of the run
                currentMax  = 1
                # Initialise the variable containing the previous zone of a comparison
                # with the first zone of the trajectory
                oldZone = self.clustersSequence[t]
                
            else:
                # If the zone has not changed
                if newZone == oldZone:
                    # Increment the max of current run
                    currentMax = currentMax + 1
                # if the zone has changed
                else:
                    oldZone = newZone;
                    # Check if a longer run has been foung
                    if currentMax > tMax:
                        tMax = currentMax   
                    # Reinitialize current max
                    currentMax = 1
   
        return tMax
    
    # Function to find the temporal transition matrices, i.e., as set of matrices
    # defining the probability of transitioning from cluster i to cluster j, 
    # considering that we spent T time instants in cluster i. The number of 
    # temporal transition matrices is equal to the max overall time spent in 
    # a cluster.
    # INPUTS:
    # - tMax: max overall time spent in a cluster;
    # - numberTrainingData: number of training datapoint (over which we loop)
    def FindTimeTransitionMatrices(self, tMax, numberTrainingData):

        transMatsTime = []
        for i in range(tMax):
            transMatsTime.append(np.zeros((self.num_clusters,self.num_clusters)))
        
        # I take the first zone in the trajectory
        prevZone = self.clustersSequence[0]
        time = -1
        
        # loop over the number of time instants of the trajectory
        for t in range(1, numberTrainingData):
            
            # I add a time instant
            time = time + 1
            
            # New zone
            newZone = self.clustersSequence[t]
            
            if (time >= tMax):
                prevZone = newZone
                time = -1
                continue
            
            # Count the transition, if we are not at the last time instant
            # of a trajectory.
            if t not in self.startingPoints:
                    
                # If I change the zone with respect to the previous time instant(/s)
                if (prevZone != newZone):
                    
                    # I increment the corresponding value in the correct transition matrix
                    transMatsTime[time][prevZone, newZone] += 1
                    # And I update the zone value
                    prevZone = newZone
                    # I reinitialize the time
                    time = -1
                # Otherwise, if I remain in the same zone
                else:
                    
                    transMatsTime[time][prevZone, newZone] += 1;
                    
            else:
                
                # If we are here, it means we are at the beginning of a trajectory
                # I update the zone value
                prevZone = newZone
                # I reinitialize the time
                time = -1
                
        self.transMatsTime = transMatsTime
        
        # For each transition matrix
        for t in range(tMax):
            # looping over rows of current matrix
            for row in range(self.num_clusters):
                # sum of the elements of the row
                sumElementsRow = np.sum(transMatsTime[t][row, :])
                
                # to prevent division by 0
                if sumElementsRow != 0:
                    # looping over columns of current matrix
                    for column in range(self.num_clusters):
                        # normalise matrix element
                        transMatsTime[t][row, column] = \
                            transMatsTime[t][row, column]/sumElementsRow
                            
        self.transMatsTime = transMatsTime
        
        return
    
    # Function to find the maximum time spent in EACH cluster.
    def CalculateMaxClustersTime(self):

        # Total length of training data
        currLength    = self.clustersSequence.shape[0]
        
        # Where to insert the max time values
        maxClustersTime = np.zeros((self.num_clusters,))
        
        # Make the list to array
        arrayStartingPoints = np.asarray(self.startingPoints)
        
        for t in range(currLength):
            
            # Zone at current time
            newZone = self.clustersSequence[t]
            
            if t in arrayStartingPoints:
                # Initialise the length of the run
                currentMax  = 1
                # Initialise the variable containing the previous zone of a comparison
                # with the first zone of the trajectory
                oldZone = self.clustersSequence[t]
            else:
                #if newZone != finalZone:
                # If the zone has not changed
                if newZone == oldZone:
                    # Increment the max of current run
                    currentMax = currentMax + 1
                # if the zone has changed
                else:
                    oldZone = newZone;
                    # Reinitialize current max
                    currentMax = 1     
                # Check if a longer run has been found
                if currentMax > maxClustersTime[oldZone]:
                    maxClustersTime[oldZone] = int(currentMax)

        self.maxClustersTime = maxClustersTime
        
        return
    
    # Temporal transition matrices averaged over a window to avoid overfitting.
    # INPUTS:
    # - window_percentage: percentage that establishes the dimension of the 
    #                      window. It is the percentage w.r.t. the max time 
    #                      spent in a cluster.
    def CalculateWindowedTemporalTransitionMatrices(self, window_percentage = 0.25):
        
        # Total length of training data
        numberTrainingData    = self.clustersSequence.shape[0]
        # Find max number of time instants before a zone change 
        tMax                  = self.FindMaxTimeBeforeTransition()
        # Maximum time spent in each cluster
        maxClustersTime       = self.maxClustersTime
        
        # Initialize
        windowedtransMatsTime = []
        for i in range(tMax):
            windowedtransMatsTime.append(np.zeros((self.num_clusters,
                                                   self.num_clusters)))
        
        # I take the first zone in the trajectory
        prevZone              = self.clustersSequence[0]
        time = -1
        
        # loop over the number of time instants of the trajectory
        for t in range(1, numberTrainingData):
            
            # I add a time instant
            time = time + 1
            
            # New zone
            newZone                  = self.clustersSequence[t]
            
            # I increment the corresponding value in the correct transition matrices
            # I do this over a window, so first I have to find the window
            maxTimeForCurrentCluster = maxClustersTime[prevZone]
            beginWindow              = time - maxTimeForCurrentCluster*window_percentage
            beginWindow              = int(np.floor(max(0, beginWindow))) # avoid going negative
            endWindow                = time + maxTimeForCurrentCluster*window_percentage
            endWindow                = int(np.floor(min(maxTimeForCurrentCluster, endWindow)))
            
            # I also have to look at what are the closest trajectories starting
            # points (the previous one and the next one), in order not to 
            # have a window that considers two separate trajectories 
            currentTrajBeginning = self.FindCurrentTrajectoryBeginningTime(t)
            currentTrajEnd       = self.FindCurrentTrajectoryEndingTime(t)
            beginWindow          = int(max(beginWindow, t-currentTrajBeginning+1))
            endWindow            = int(min(endWindow  , currentTrajEnd-t-1))
            
            # Count the transition, if we are not at the last time instant
            # of a trajectory.
            if t not in self.startingPoints:
                for w in range(beginWindow, endWindow):
                    windowedtransMatsTime[w][prevZone, newZone] += 1
            
            # If I change the zone with respect to the previous time instant(/s)
            if (prevZone != newZone):
                # And I update the zone value
                prevZone = newZone
                # I reinitialize the time
                time = -1
        
        # For each transition matrix
        for t in range(tMax):
            # looping over rows of current matrix
            for row in range(self.num_clusters):
                # sum of the elements of the row
                sumElementsRow = np.sum(windowedtransMatsTime[t][row, :])
                
                # to prevent division by 0
                if sumElementsRow != 0:
                    # looping over columns of current matrix
                    for column in range(self.num_clusters):
                        # normalise matrix element
                        windowedtransMatsTime[t][row, column] = \
                            windowedtransMatsTime[t][row, column]/sumElementsRow
                            
        self.windowedtransMatsTime = windowedtransMatsTime
            
        return
    
    ###########################################################################
    # Functions to created some extended versions of the transition matrices
    
    # Function to create an extended transition matrix that includes also the
    # clusters that are very near each cluster.
    def CalculateTransitionMatrixExtendedThroughVicinityMatrix(self):
        
        transitionMatExploration = torch.zeros(self.num_clusters, self.num_clusters) + 1e-30
        
        for i in range(self.num_clusters):
            
            for j in range(self.num_clusters):
                
                if self.vicinityTransitionMat[i,j] == 1 and i != j:
                    
                    transitionMatExploration[i,:] += self.transitionMat[j,:]
                    
        # Normalize the rows
        for i in range(self.num_clusters):
            
            # If there is at least one translation on the row
            if torch.sum(transitionMatExploration[i,:]) > 0.5:
                transitionMatExploration[i,:] = transitionMatExploration[i,:]/torch.sum(transitionMatExploration[i,:])
            
        self.transitionMatExploration = transitionMatExploration
                    
        return
    
    ###########################################################################
    # Functions for initializing empty graph
    
    # Initialize the clustering with two nodes
    # input: - z_dim: dimension of latent states
    #        - low_value_z, low_value_z_dev (optional): min values of 
    #          z and of its derivative
    #        - high_value_z, high_value_z_dev (optional): max values of 
    #          z and of its derivative
    def __init__(self, 
                 z_dim, 
                 low_value_z  = -1, low_value_z_dev  = -0.01, 
                 high_value_z =  1, high_value_z_dev =  0.01):
        
        # Initial 2 nodes for training the algorithm
        self.num_clusters = 2                                                                    
        
        # Mean of nodes
        self.nodesMean = np.zeros((self.num_clusters, z_dim*2))
        
        # It returns an array of random numbers generated from the continuous uniform 
        for i in range(self.num_clusters):
            for j in range(z_dim*2):
                
                if j < z_dim:
                    self.nodesMean[i,j] = np.random.uniform(low = low_value_z    , high = high_value_z    , size=None)   
                    
                else:
                    self.nodesMean[i,j] = np.random.uniform(low = low_value_z_dev, high = high_value_z_dev, size=None) 
                    
        # Don't put any clusters sequence yet
        self.clustersSequence = 0                           

        return
    
    # Function to inizialize to zero the graph connections (to use at beginning
    # of GNG training).
    def InitializeGraphConnections(self):
        
        # error
        self.E       = np.zeros(self.num_clusters)
        # utility
        self.utility = np.ones(self.num_clusters)
        # Connections between nodes
        self.C       = np.zeros((self.num_clusters, self.num_clusters))
        # Ages of the edges
        self.t       = np.zeros((self.num_clusters, self.num_clusters))
        
        return
    
    ###########################################################################
    # Function for loading a graph having data and the sequence of cluster
    # assignments.
    
    # Initialize the clustering knowing which are the datapoint and which
    # are the cluster assignments of the datapoints, i.e.,:
    # - data
    # - clustersSequence
    def LoadGraphFromDataAndAssignments(self, data, clustersSequence):
        
        self.num_clusters     = int(max(clustersSequence) - min(clustersSequence)) + 1
        
        firstClusterIndex     = np.min(clustersSequence)
        clustersSequence      = clustersSequence - firstClusterIndex
        clustersSequence      = clustersSequence.astype(int)
        self.clustersSequence = np.squeeze(clustersSequence)
        
        dimensionState        = data.shape[1]
        
        # Define one array containing the data for each cluster:
        datanodes = self.FindDataForEachCluster(data)  
        self.datanodes = datanodes
        
        # Find the mean of each cluster
        self.FindMeanForEachCluster(datanodes, dimensionState)
        
        # Find the covariance for each cluster
        self.FindCovarianceForEachCluster(datanodes, dimensionState)
        
        # Find the transition matrices
        self.CalculateAllTransitionMatrices()
        
        # Find nodes distance and nodes vicinity matrices
        #self.FindNodesDistanceMatrix()
        #self.FindNodesVicinityMatrix()
        
        return
    
    ###########################################################################
    # Functions for finding data belonging to each cluster, covariance of
    # clusters and mean of clusters.
    
    # Function to find the datanodes variable (data in each cluster)
    # INPUTS:
    # - data: clustering data,
    # OUTPUTS:
    # - datanodes: list in which each element contains all the data
    #   belonging to a cluster.
    def FindDataForEachCluster(self, data):
        
        lengthOfData = data.shape[0]
        
        # Define one array for each cluster:
        datanodes = []
        for i in range(self.num_clusters):
            cluster_i = []
            datanodes.append(cluster_i)
            
        # Insert the data of each cluster in the corresponding array
        for i in range(lengthOfData):
            superstate_i = int(self.clustersSequence[i])
            state_i = data[i, :]
            datanodes[superstate_i].append(state_i)
            
        return datanodes
    
    # Function to find the covariance of each cluster.
    # INPUTS:
    # - datanodes: list in which each element contains all the data
    #   belonging to a cluster.
    # - dimensionState: dimension of the state.
    def FindCovarianceForEachCluster(self, datanodes, dimensionState):
        
        # Find the covariance for each cluster
        nodesCov = []
        for i in range(self.num_clusters):
            nodesCov_i = []
            nodesCov.append(nodesCov_i)
            
        # Looping over the number of clusters
        for i in range(self.num_clusters):
            # If there is no data in that node
            if len(datanodes[i]) == 0:
                nodesCov[i].append(np.zeros((dimensionState, dimensionState)))
            # If there is data in the node
            else:        
                datanode_of_cluster = np.asarray(datanodes[i])
                nodesCov[i] = np.cov(np.transpose(datanode_of_cluster))
                
        self.nodesCov = nodesCov
        self.nodesCov = np.stack(self.nodesCov)
        
        return
    
    # Function to find the mean of each cluster.
    # INPUTS:
    # - datanodes: list in which each element contains all the data
    #   belonging to a cluster.
    # - dimensionState: dimension of the state.
    def FindMeanForEachCluster(self, datanodes, dimensionState):
        
        # Find the covariance for each cluster
        nodesMean = np.zeros((self.num_clusters, dimensionState))
            
        # Looping over the number of clusters
        for i in range(self.num_clusters):
            # If there is no data in that node
            if len(datanodes[i]) == 0:
                nodesMean[i,:]      = np.zeros(dimensionState)
            # If there is data in the node
            else:        
                datanode_of_cluster = np.asarray(datanodes[i])
                print(datanode_of_cluster.shape)
                nodesMean[i,:]      = np.mean(datanode_of_cluster, axis = 0)
                
        self.nodesMean = nodesMean
        
        return
    
    ###########################################################################   
    # Function to save a graph as a matlab structure
    def SaveGraphToMATLAB(self, fileName):
        
        graphToSave = {
                'num_clusters'          : self.num_clusters,
                'nodesMean'             : self.nodesMean,
                'nodesCov'              : self.nodesCov,
                'transitionMat'         : self.transitionMat,
                'transMatsTime'         : self.transMatsTime,
                'windowedtransMatsTime' : self.windowedtransMatsTime,
                'clustersSequence'      : self.clustersSequence,
                'maxClustersTime'       : self.maxClustersTime
                }
        
        savemat(fileName, {'graph': graphToSave})
        
        return
    
    ###########################################################################   
    # A PRIVATE function to smoothen the cluster assignments.
    # Don't use this outside, as otherwise cluster assignment property and
    # transition matrices properties will not be coherent w.r.t. each other 
    # any more!!
    def __SmoothClusterAssignments(self, deleteThreshold = 0):
        
        # Bring to numpy to execute calculation
        flagTorch = False
        if type(self.clustersSequence) == torch.Tensor: 
            flagTorch             = True
            self.clustersSequence = self.clustersSequence.detach().cpu().numpy()   
            
        # Create array with same elements as original one
        clustersSequenceSmoothed  = copy.deepcopy(self.clustersSequence)
        
        # Number of data points in training
        dataLength = len(clustersSequenceSmoothed)
        
        # Looping over the data points
        for i in range(deleteThreshold, dataLength - (2*deleteThreshold)):

            currentCluster = self.clustersSequence[i]
            
            # Define window for cluster substitution
            windowBegin = i - deleteThreshold
            windowEnd   = i + 2*deleteThreshold - 1
            
            # Counting how many times the cluster appears in the window
            howManyTimesClusterAppearsInTheWindow = np.sum(self.clustersSequence[windowBegin:windowEnd] == currentCluster)
            
            if howManyTimesClusterAppearsInTheWindow > deleteThreshold:
                # If there are at least a number n = delete_threshold of that same
                # cluster assignment in the window, 
                # do nothing.
                dummy = 0
            else:
                # Find the other cluster that is present
                otherClustersIndex   = np.nonzero(self.clustersSequence[windowBegin:windowEnd] != currentCluster)[0] 
                
                # Try with each cluster appearing, to see if it appears more than the original one
                # and more than the other ones
                
                # As first most-appearing cluster, we take the original one
                frequencyOfClusterAppearingMost = howManyTimesClusterAppearsInTheWindow
                clusterAppearingMost            = currentCluster
                
                for j in range(len(otherClustersIndex)):
                    
                    # Cluster
                    otherCluster = self.clustersSequence[windowBegin-1+otherClustersIndex[j]]
                    # How many times it appears in the window
                    howManyTimesClusterAppearsInTheWindow = np.sum(self.clustersSequence[windowBegin:windowEnd] == otherCluster)
                    
                    # Is it more frequent?
                    if howManyTimesClusterAppearsInTheWindow > frequencyOfClusterAppearingMost:
                        frequencyOfClusterAppearingMost = howManyTimesClusterAppearsInTheWindow
                        clusterAppearingMost            = otherCluster
                        
                    # Substitute cluster
                    clustersSequenceSmoothed[i] = clusterAppearingMost
        
        # Bring back to torch if data was in torch
        if flagTorch == True:
            clustersSequenceSmoothed = torch.from_numpy(clustersSequenceSmoothed).float()
                    
        # Save sequence of data
        self.clustersSequence = clustersSequenceSmoothed
  
        return
    
    # Smooths the cluster assignments and recalculates all the properties consequently.
    # INPUTS:
    # - deleteThreshold: if a cluster lasts less than this threshold, it is 
    #                    eliminated;
    # OUTPUTS:
    # - vectorOfKeptClusters: Vector with values equal to either 0 or 1 (0 if the cluster is 
    #                    eliminated, 1 if it is kept).
    def SmoothClusterAssignmentsAndRecalculateProperties(self, deleteThreshold = 0):
        
        if deleteThreshold > 0:
        
            # Smoothen cluster assignments
            self.__SmoothClusterAssignments(deleteThreshold)
            # Recalculate ALL the transition matrices
            self.CalculateAllTransitionMatrices()
            
            # Vector with values equal to either 0 or 1 (0 if the cluster is 
            # eliminated, 1 if it is kept).
            vectorOfKeptClusters = np.ones(self.num_clusters)
            
            # Are there any nan values in transition matrix? i.e., clusters that
            # were eliminated?
            countOverClusters = 0
            for i in range(self.num_clusters):
                if type(self.transitionMat)   == np.ndarray:
                    if np.isnan(self.transitionMat[countOverClusters,0]):
                        print(countOverClusters)
                        self.EliminateOneCluster(countOverClusters)
                        countOverClusters -= 1
                        vectorOfKeptClusters[i] = 0
                elif type(self.transitionMat) == torch.Tensor:
                    if torch.isnan(self.transitionMat[countOverClusters,0]):
                        self.EliminateOneCluster(countOverClusters)
                        countOverClusters -= 1
                        vectorOfKeptClusters[i] = 0
                countOverClusters += 1
        else:
            vectorOfKeptClusters = np.ones(self.num_clusters)
            
        # Recalculate the nodes distance and vicinity matrices
        #self.FindNodesDistanceMatrix()
        #self.FindNodesVicinityMatrix()
        
        return vectorOfKeptClusters
    
    ###########################################################################   
    # Functions to eliminate clusters from a graph. The functions to smoothen
    # the cluster assignments use these.
    
    # Static function to eliminate a row/column from a transition matrix
    # INPUTS:
    # transitionMatrix: transition matrix from which to eliminate a row/column
    # indexOfRowAndColumn: index of row/column to eliminate
    # OUTPUTS:
    # transitionMatrix: transition matrix from which we eliminated a row/column
    @staticmethod
    def EliminateRowAndColumnFromMatrix(transitionMatrix, indexOfRowAndColumn):
        
        if type(transitionMatrix)   == np.ndarray:
            transitionMatrix = np.delete(transitionMatrix, indexOfRowAndColumn, axis=0)
            transitionMatrix = np.delete(transitionMatrix, indexOfRowAndColumn, axis=1)           
        elif type(transitionMatrix) == torch.Tensor:
            allIndicesExceptOneToEliminate = torch.arange(transitionMatrix.size(0))!=indexOfRowAndColumn
            transitionMatrix = transitionMatrix[allIndicesExceptOneToEliminate, :] 
            transitionMatrix = transitionMatrix[:, allIndicesExceptOneToEliminate] 
        
        return transitionMatrix
    
    # Function to eliminate a row from a nodesMean matrix
    # INPUTS:
    # nodesMean: nodesMean matrix from which to eliminate a row
    # indexOfRowAndColumn: index of row to eliminate
    # OUTPUTS:
    # nodesMean: nodesMean matrix from which we eliminated a row
    @staticmethod
    def EliminateRowFromMatrix(nodesMean, indexOfRow):
        
        if type(nodesMean)   == np.ndarray:
            nodesMean = np.delete(nodesMean, indexOfRow, axis=0)  
        elif type(nodesMean) == torch.Tensor:
            allIndicesExceptOneToEliminate = torch.arange(nodesMean.size(0))!=indexOfRow
            nodesMean = nodesMean[allIndicesExceptOneToEliminate, :] 
        
        return nodesMean
    
    # Function to eliminate a value from a vector
    # INPUTS:
    # vector: vector from which to eliminate a value
    # index: index of value to eliminate
    # OUTPUTS:
    # vector: vector from which we eliminated a value
    @staticmethod
    def EliminateValueFromVector(vector, index):
        
        if type(vector)   == np.ndarray:
            vector = np.delete(vector, index, axis=0)  
        elif type(vector) == torch.Tensor:
            allIndicesExceptOneToEliminate = torch.arange(vector.size(0))!=index
            vector = vector[allIndicesExceptOneToEliminate] 
        
        return vector
    
    # Function to eliminate a value from a clusters sequence
    # INPUTS:
    # - index: index of cluster to eliminated
    def EliminateClusterFromClustersSequence(self, index):
        
        for i in range(len(self.clustersSequence)):
            
            currentCluster = self.clustersSequence[i] 
            
            if currentCluster >= index:
                self.clustersSequence[i] = currentCluster - 1
        
        return
    
    # If we want to eliminate one cluster from the graph
    # (for example, after smoothing, there is a cluster that is not used
    # any more)
    # INPUTS:
    # - indexOfCluster: index of cluster to eliminated
    def EliminateOneCluster(self, indexOfCluster):
        
        # Number of clusters
        self.num_clusters    = self.num_clusters - 1
        
        # From NodesMean
        self.nodesMean       = ClusteringGraph.EliminateRowFromMatrix(self.nodesMean, indexOfCluster)
        # From NodesCov
        self.nodesCov        = ClusteringGraph.EliminateRowFromMatrix(self.nodesCov, indexOfCluster)
        # From Transition matrix
        self.transitionMat   = ClusteringGraph.EliminateRowAndColumnFromMatrix(self.transitionMat, indexOfCluster)
        
        # From temporal matrices - related values
        newMaxClustersTime = ClusteringGraph.EliminateValueFromVector(self.maxClustersTime, indexOfCluster)
        maxTimeOverall     = np.max(newMaxClustersTime)
        
        # Temporal transition matrices
        for i in range(int(np.max(self.maxClustersTime))):
            if i <= maxTimeOverall:
                print('HERE')
                print(i)
                print(indexOfCluster)
                print(len(self.transMatsTime))
                print(self.transMatsTime[i].shape)
                self.transMatsTime[i] = ClusteringGraph.EliminateRowAndColumnFromMatrix(self.transMatsTime[i], indexOfCluster)
            else:
                self.transMatsTime.pop(i)
                
        # Windowed temporal transition matrices
        for i in range(len(self.windowedtransMatsTime)):
            self.windowedtransMatsTime[i] = ClusteringGraph.EliminateRowAndColumnFromMatrix(self.windowedtransMatsTime[i], indexOfCluster)
        
        # From temporal matrices - related values
        self.maxClustersTime = newMaxClustersTime
        
        # From clusters assignment
        self.EliminateClusterFromClustersSequence(indexOfCluster)
        
        #######################################################################
        # Additional features to the base one      
        if self.nodesCovPred != None:
            self.nodesCovPred.pop(indexOfCluster)
        if self.nodesCovD != None:
            self.nodesCovD.pop(indexOfCluster)
           
        return
    
    ###########################################################################
    # Function to create the graph interactions 
    # (To use in Graph Matching code).
    # Needs as input self.clustersSequence.
    # Outputs: - index: key of clusters
    #          - interaction: list of transitions between clusters
    def FindInteractionsList(self):
        
        # Create the index list: it just gives a correspondence key-index
        index = dict()
        for i in range (0, self.num_clusters):
            index[i] = i
           
        # list with all the interactions from a node i to a node j, with i != j
        interactions = []
        for i in range (0, len(self.clustersSequence) - 1):
            # If we have a movement from a cluster to another one
            #if idx[i + 1] != idx[i]:
            # Add the interaction to the list
            pair = [self.clustersSequence[i + 1], self.clustersSequence[i]]
            interactions.append(pair)
        # Sort the list in ascending order of numbers
        interactions.sort() 
        
        self.indicesList      = index
        self.interactionsList = interactions
        
        return
    
    ###########################################################################
    # Functions that bring the variables in the object from numpy to torch.
    
    def BringGraphToTorch(self):        
        self.nodesMean            = torch.from_numpy(self.nodesMean).to(device)
        self.clustersSequence     = torch.from_numpy(self.clustersSequence).to(device)
        self.nodesCov         = torch.from_numpy(self.nodesCov).to(device)
        for i in range(len(self.transMatsTime)):
            self.transMatsTime[i] = torch.from_numpy(self.transMatsTime[i]).to(device)
        for i in range(len(self.transMatsTime)):
            self.windowedtransMatsTime[i] = torch.from_numpy(self.windowedtransMatsTime[i]).to(device)
            
        if hasattr(self, 'additionalClusterInfo'):
            self.additionalClusterInfo = torch.from_numpy(self.additionalClusterInfo).to(device)
            
        if hasattr(self, 'nodesDistanceMatrix'):
            self.nodesDistanceMatrix   = torch.from_numpy(self.nodesDistanceMatrix).to(device)

        self.BringTransitionMatrixToTorch()
        
        #######################################################################
        # Additional features to the base one      
        if self.nodesCovPred != None:
            for i in range(self.num_clusters):
                self.nodesCovPred[i]   = torch.from_numpy(self.nodesCovPred[i]).to(device)
        if self.nodesCovD != None:
            for i in range(self.num_clusters):
                self.nodesCovD[i]      = torch.from_numpy(self.nodesCovD[i]).to(device)
        
        return
        
    
    def BringTransitionMatrixToTorch(self):
        self.transitionMat = torch.from_numpy(self.transitionMat).to(device)
        return
    
    ###########################################################################
    # Print function
    
    def print(self):
        
        print('Number of clusters in graph:')
        print(self.num_clusters)
        print('Sequence of clusters')
        print(self.clustersSequence)
        #print('Cluster means in graph:')
        #print(self.nodesMean)
        #print('Synchronized data in graph:')
        #print(self.data_m)
        #print('Cluster covariances in graph:')
        #print(self.nodesCov)
        
        return
    
    ###########################################################################
    # Functions to calculate the distances of datapoint from the clusters of the graph.
            
    # Find the distance of a set of points from each node of a cluster graph
    # inputs: - points_sequence: sequence of odometry/distance points
    #                            The form of this data should be:
    #                            > dimension 0: number of sequences;
    #                            > dimension 1: length of sequence (fixed len);
    #                            > dimension 2: dimension of sequence (e.g., if pos x, pos y = 2).
    #         - obsCovariance: covariance of observation
    # output: distances of each datapoint in the sequences from each cluster. 
    #                            The form of this data will be:
    #                            > dimension 0: number of sequences;
    #                            > dimension 1: length of sequence (fixed len);
    #                            > dimension 2: number of clusters (num_clusters).
    def FindBhattaDistancesFromSequencesToClusters(self, points_sequence, obsCovariance):
        
        # Information about the sequence of points
        num_sequences        = points_sequence.shape[0] # how many sequences
        num_time_in_sequence = points_sequence.shape[1] # length of sequence
        
        # Calculating the distances
        if type(points_sequence)   == np.ndarray:
            distances     = np.zeros((num_sequences,num_time_in_sequence,self.num_clusters))
            obsVariance = np.diag(obsCovariance)
        elif type(points_sequence) == torch.Tensor:
            distances     = torch.zeros(num_sequences,num_time_in_sequence,self.num_clusters).to(device)
            obsVariance = torch.diag(obsCovariance).to(device)
            
        if type(points_sequence)   == np.ndarray:
            diagonalizedNodesCov = np.diagonal(self.nodesCov, axis1 = 1, axis2 = 2)
        elif type(points_sequence) == torch.Tensor:
            diagonalizedNodesCov = torch.diagonal(self.nodesCov, dim1 = 1, dim2 = 2)

        # looping over the number of sequences
        for i in range(num_sequences):
            # Looping over the number of elements in a sequence
            for j in range(num_time_in_sequence):
                # current measurement
                measurement = points_sequence[i,j,:]
                
                # Calculating Bhattacharya distance between observation and clusters
                if type(points_sequence)   == np.ndarray:
                    currDistances   = d_utils.CalculateBhattacharyyaDistance(measurement, obsVariance, self.nodesMean, diagonalizedNodesCov)
                elif type(points_sequence) == torch.Tensor:
                    currDistances   = d_utils.CalculateBhattacharyyaDistanceTorch(measurement, obsVariance, self.nodesMean, diagonalizedNodesCov)
                    
                distances[i,j,:] = currDistances
                    
        return distances
    
    # Find the distance of a single point from each node of a cluster graph.
    # The diagonal value of the covariance of the point is given.
    # INPUTS:
    # point: datapoint's mean.
    # variance:  diagonal value of the covariance of the point
    # OUTPUTS:
    # distances: distances of the point from each of the clusters of the graph.
    def FindBhattaDistanceFromClustersSinglePointGivenVariance(self, point, variance):
        
        # Calculating the distances
        if type(point)   == np.ndarray:
            distances     = np.zeros(self.num_clusters)
        elif type(point) == torch.Tensor:
            distances     = torch.zeros(self.num_clusters).to(device)
            
        if type(point)   == np.ndarray:
            diagonalizedNodesCov = np.diagonal(self.nodesCov, axis1 = 1, axis2 = 2)
        elif type(point) == torch.Tensor:
            diagonalizedNodesCov = torch.diagonal(self.nodesCov, dim1 = 1, dim2 = 2)

        # current measurement
        measurement = point
        
        # Calculating Bhattacharya distance between observation and clusters
        if type(point)   == np.ndarray:
            distances   = d_utils.CalculateBhattacharyyaDistance(measurement, variance, self.nodesMean, diagonalizedNodesCov)
        elif type(point) == torch.Tensor:
            distances   = d_utils.CalculateBhattacharyyaDistanceTorch(measurement, variance, self.nodesMean, diagonalizedNodesCov)

        return distances
    
    # Find the distance of a single point from each node of a cluster graph.
    # The covariance of the point is given.
    # INPUTS:
    # point: datapoint's mean.
    # variance:  the covariance of the point
    # OUTPUTS:
    # distances: distances of the point from each of the clusters of the graph.
    def FindBhattaDistanceFromClustersSinglePointGivenCovariance(self, point, covariance):
        
        # Calculating the distances
        if type(point)   == np.ndarray:
            distances     = np.zeros(self.num_clusters)
            obsVariance   = np.diagonal(covariance)
        elif type(point) == torch.Tensor:
            distances     = torch.zeros(self.num_clusters).to(device)
            obsVariance   = torch.diagonal(covariance).to(device)
            
        if type(point)   == np.ndarray:
            diagonalizedNodesCov = np.diagonal(self.nodesCov, axis1 = 1, axis2 = 2)
        elif type(point) == torch.Tensor:
            diagonalizedNodesCov = torch.diagonal(self.nodesCov, dim1 = 1, dim2 = 2)

        # current measurement
        measurement = point
        
        # Calculating Bhattacharya distance between observation and clusters
        if type(point)   == np.ndarray:
            distances   = d_utils.CalculateBhattacharyyaDistance(measurement, obsVariance, self.nodesMean, diagonalizedNodesCov)
        elif type(point) == torch.Tensor:
            distances   = d_utils.CalculateBhattacharyyaDistanceTorch(measurement, obsVariance, self.nodesMean, diagonalizedNodesCov)

        return distances
    
    # Different to the previous one: we suppose to have a single sequence of length 1, so there
    # is no need for the two 'for' loops.
    def FindBhattaDistancesFromSequencesToClustersSinglePoint(self, points_sequence, obsCovariance):
        
        # Information about the sequence of points
        num_sequences        = points_sequence.shape[0] # how many sequences
        num_time_in_sequence = points_sequence.shape[1] # length of sequence
        
        # Calculating the distances
        if type(points_sequence)   == np.ndarray:
            distances     = np.zeros((num_sequences,num_time_in_sequence,self.num_clusters))
            obsVariance = np.diagonal(obsCovariance)
        elif type(points_sequence) == torch.Tensor:
            distances     = torch.zeros(num_sequences,num_time_in_sequence,self.num_clusters).to(device)
            obsVariance = torch.diagonal(obsCovariance).to(device)

        # current measurement
        measurement = points_sequence[0,0,:]
        
        if type(points_sequence)   == np.ndarray:
            diagonalizedNodesCov = np.diagonal(self.nodesCov, axis1 = 1, axis2 = 2)
        elif type(points_sequence) == torch.Tensor:
            diagonalizedNodesCov = torch.diagonal(self.nodesCov, dim1 = 1, dim2 = 2)
        
        # Calculating Bhattacharya distance between observation and clusters
        for index_s in range(self.num_clusters):
            
            # Calculating Bhattacharya distance between observation and clusters
            if type(points_sequence)   == np.ndarray:
                currDistances   = d_utils.CalculateBhattacharyyaDistance(measurement, obsVariance, self.nodesMean, diagonalizedNodesCov)
            elif type(points_sequence) == torch.Tensor:
                currDistances   = d_utils.CalculateBhattacharyyaDistanceTorch(measurement, obsVariance, self.nodesMean, diagonalizedNodesCov)
                
            distances[0,0,:] = currDistances
                    
        return distances
    
    def FindDistancesFromVideoClustersAbsOfMeans(self, videoStateData):
        
        # Calculating the distances
        distances = torch.mean(torch.abs(videoStateData - self.nodesMean), 2)      
        distances = torch.unsqueeze(distances, 0)     
        
        return distances
    
    # This function finds the distances of a sequence of data from the 
    # mean of the clusters, using the VIDEO clusters and so the ENCODED STATE 'a'
    # obtained after encoding with the VAE.
    # INPUTS:
    # - videoStateData: the encoded state 'a' of the data
    # OUTPUTS:
    # - the distances of each encoded state 'a' from the K video cluster centers.
    def FindDistancesFromVideoClustersNoCov(self, videoStateData):
        
        # Define an observation covariance for each 
        dim_sequences     = videoStateData.shape[2] # state dimension
        
        r1                = 1e-8
        if type(videoStateData)   == np.ndarray:
            obsCovariance = np.eye(dim_sequences)*r1
        elif type(videoStateData) == torch.Tensor:
            obsCovariance = (torch.eye(dim_sequences)*r1).to(device)
        
        # Extract the Bhattacharya distances of the datapoints from the 
        # clusters of the graph
        distances         = self.FindBhattaDistancesFromSequencesToClustersSinglePoint(
                                               videoStateData, obsCovariance) 
        return distances
    
    # This function finds the distances of a sequence of data from the 
    # mean of the clusters, using the ODOMETRY clusters
    def FindDistancesFromParamsClustersNoCov(self, paramsStateData):
        
        # paramsStateData should have the structure:
        # (number of sequences, length of sequence, dimension of state)
        # so if it smaller, probably it only has the state dimension or
        # length of sequence one, so we add further empty dimension on top
        if paramsStateData.dim() == 2:
            paramsStateData = torch.unsqueeze(paramsStateData, 0)
        elif paramsStateData.dim() == 1:
            paramsStateData = torch.unsqueeze(paramsStateData, 0)
            paramsStateData = torch.unsqueeze(paramsStateData, 0)
            
        # Define an observation covariance for each 
        dim_sequences     = paramsStateData.shape[2] # state dimension
            
        r1                = 1e-18
        if type(paramsStateData)   == np.ndarray:
            obsCovariance = np.eye(dim_sequences)*r1
        elif type(paramsStateData) == torch.Tensor:
            obsCovariance = (torch.eye(dim_sequences)*r1).to(device)
        
        # Extract the Bhattacharya distances of the datapoints from the 
        # clusters of the graph
        distances         = self.FindBhattaDistancesFromSequencesToClustersSinglePoint(
                                               paramsStateData, obsCovariance)
        return distances
    
    ###########################################################################
    # Functions that calculate the distance matrices between nodes of the 
    # graph.
    
    # Function to find the distance between nodes of the graph
    def FindNodesDistanceMatrix(self):
    
        nodesDistanceMatrix = np.zeros((self.num_clusters,self.num_clusters))
        
        # For source graph
        for i in range (0, self.num_clusters):
            for j in range (0, self.num_clusters):
                # node i
                mean_a = self.nodesMean[i, :]
                mean_b = self.nodesMean[j, :]
                
                distance_ab = np.mean((mean_a - mean_b)**2)
                
                nodesDistanceMatrix[i, j] = distance_ab
    
        self.nodesDistanceMatrix = nodesDistanceMatrix
    
        return
    
    # This function generates a distance matrix using Bhattacharya distance
    def FindNodesBhattacharyaDistanceMatrix(self):
        
        nodesDistanceMatrix = np.zeros((self.num_clusters,self.num_clusters))
        
        # For source graph
        for i in range (0, self.num_clusters):
            for j in range (0, self.num_clusters):
                
                # means of nodes
                mean_a = self.nodesMean[i, :]
                mean_b = self.nodesMean[j, :]
                # covariances of nodes
                cov_a  = self.nodesCov[i]
                cov_b  = self.nodesCov[j]
                # variances of nodes
                var_a  = np.diag(cov_a)
                var_b  = np.diag(cov_b)
                
                # Bhattacharya distance calculation
                distance_ab = d_utils.CalculateBhattacharyyaDistance(mean_a, var_a, mean_b, var_b)
                
                # Inserting the distance in the matrix
                nodesDistanceMatrix[i, j] = distance_ab
    
        self.nodesDistanceMatrix = nodesDistanceMatrix
        
        return
    
    
###########################################################################
###########################################################################
# TEST CLASS
                
class TestsClusteringGraph(unittest.TestCase):   
    
    def _InitializeDummyClusteringGraph(self):
        
        self.clusteringGraph = ClusteringGraph(z_dim = 2)        
        return                
        
    def CheckFindTransitionMatrix(self):
        
        # Initialization
        self._InitializeDummyClusteringGraph()
        self.clusteringGraph.num_clusters = 5
        self.clusteringGraph.clustersSequence = \
           np.array([0,0,0,0,1,1,1,1,1,1,0,0,0,2,2,2,2,2,2,2,3,3,3,3,3,3,4,0,0,0,0,0])
        self.clusteringGraph.startingPoints = []
        self.clusteringGraph.startingPoints.append(0) 
        
        # Find transition matrix
        self.clusteringGraph.FindTransitionMatrix()        
        # Correct transition matrix
        correctTransitionMatrix = np.array([[9/11, 1/11, 1/11, 0  , 0  ],
                                            [1/6 , 5/6 , 0   , 0  , 0  ],
                                            [0   , 0   , 6/7 , 1/7, 0  ],
                                            [0   , 0   , 0   , 5/6, 1/6],
                                            [1   , 0   , 0   , 0  , 0  ]])
    
        self.assertTrue((correctTransitionMatrix == self.clusteringGraph.transitionMat).all())           
        return
    
    def CheckFindTransitionMatrix_multipleStartingPoints(self):
        
        # Initialization
        self._InitializeDummyClusteringGraph()
        self.clusteringGraph.num_clusters = 5
        self.clusteringGraph.clustersSequence = \
           np.array([0,0,0,0,1,1,1,1,1,1,0,0,0,2,2,2,2,2,2,2,3,3,3,3,3,3,4,0,0,0,0,0])
        self.clusteringGraph.startingPoints = []
        self.clusteringGraph.startingPoints.append(0) 
        self.clusteringGraph.startingPoints.append(6) 
        self.clusteringGraph.startingPoints.append(23) 
        
        # Find transition matrix
        self.clusteringGraph.FindTransitionMatrix()        
        # Correct transition matrix
        correctTransitionMatrix = np.array([[9/11, 1/11, 1/11, 0  , 0  ],
                                            [1/5 , 4/5 , 0   , 0  , 0  ],
                                            [0   , 0   , 6/7 , 1/7, 0  ],
                                            [0   , 0   , 0   , 4/5, 1/5],
                                            [1   , 0   , 0   , 0  , 0  ]])
    
        self.assertTrue((correctTransitionMatrix == self.clusteringGraph.transitionMat).all())           
        return
    
    def CheckCalculateTemporalTransitionMatrices_multipleStartingPoints(self):
        
        # Initialization
        self._InitializeDummyClusteringGraph()
        self.clusteringGraph.num_clusters = 5
        self.clusteringGraph.clustersSequence = \
           np.array([0,0,1,0,0   ,0,2,2,3,3,    3,3,4,0])
        self.clusteringGraph.startingPoints = []
        self.clusteringGraph.startingPoints.append(0) 
        self.clusteringGraph.startingPoints.append(5) 
        self.clusteringGraph.startingPoints.append(10) 
        
        # Find transition matrix
        self.clusteringGraph.CalculateTemporalTransitionMatrices()  
        # Correct transition matrix
        correctTemporalMatrix1 = np.array([[2/3 , 0   , 1/3 , 0  , 0  ],
                                           [1   , 0   , 0   , 0  , 0  ],
                                           [0   , 0   , 1   , 0  , 0  ],
                                           [0   , 0   , 0   , 1  , 0  ],
                                           [1   , 0   , 0   , 0  , 0  ]])
        correctTemporalMatrix2 = np.array([[0   , 1   , 0   , 0  , 0  ],
                                           [0   , 0   , 0   , 0  , 0  ],
                                           [0   , 0   , 0   , 1  , 0  ],
                                           [0   , 0   , 0   , 0  , 1  ],
                                           [0   , 0   , 0   , 0  , 0  ]])

        self.assertTrue((correctTemporalMatrix1 == self.clusteringGraph.transMatsTime[0]).all()) 
        self.assertTrue((correctTemporalMatrix2 == self.clusteringGraph.transMatsTime[1]).all()) 
        return
    
    def CheckCalculateTemporalTransitionMatrices(self):
        
        # Initialization
        self._InitializeDummyClusteringGraph()
        self.clusteringGraph.num_clusters = 5
        self.clusteringGraph.clustersSequence = \
           np.array([0,0,1,0,0,0,2,2,3,3,3,3,4,0])
        self.clusteringGraph.startingPoints = []
        self.clusteringGraph.startingPoints.append(0) 
        
        # Find transition matrix
        self.clusteringGraph.CalculateTemporalTransitionMatrices()        
        # Correct transition matrix
        correctTemporalMatrix1 = np.array([[1   , 0   , 0   , 0  , 0  ],
                                           [1   , 0   , 0   , 0  , 0  ],
                                           [0   , 0   , 1   , 0  , 0  ],
                                           [0   , 0   , 0   , 1  , 0  ],
                                           [1   , 0   , 0   , 0  , 0  ]])
        correctTemporalMatrix2 = np.array([[0.5 , 0.5 , 0   , 0  , 0  ],
                                           [0   , 0   , 0   , 0  , 0  ],
                                           [0   , 0   , 0   , 1  , 0  ],
                                           [0   , 0   , 0   , 1  , 0  ],
                                           [0   , 0   , 0   , 0  , 0  ]])
        correctTemporalMatrix3 = np.array([[0   , 0   , 1   , 0  , 0  ],
                                           [0   , 0   , 0   , 0  , 0  ],
                                           [0   , 0   , 0   , 0  , 0  ],
                                           [0   , 0   , 0   , 1  , 0  ],
                                           [0   , 0   , 0   , 0  , 0  ]])
        correctTemporalMatrix4 = np.array([[0   , 0   , 0   , 0  , 0  ],
                                           [0   , 0   , 0   , 0  , 0  ],
                                           [0   , 0   , 0   , 0  , 0  ],
                                           [0   , 0   , 0   , 0  , 1  ],
                                           [0   , 0   , 0   , 0  , 0  ]])
    
        self.assertTrue((correctTemporalMatrix1 == self.clusteringGraph.transMatsTime[0]).all()) 
        self.assertTrue((correctTemporalMatrix2 == self.clusteringGraph.transMatsTime[1]).all()) 
        self.assertTrue((correctTemporalMatrix3 == self.clusteringGraph.transMatsTime[2]).all()) 
        self.assertTrue((correctTemporalMatrix4 == self.clusteringGraph.transMatsTime[3]).all()) 
        return
    
    def CheckCalculateMaxClustersTime_short(self):
        
        # Initialization
        self._InitializeDummyClusteringGraph()
        self.clusteringGraph.num_clusters = 5
        self.clusteringGraph.clustersSequence = \
           np.array([0,0,1,0,0,0,2,2,3,3,3,3,4,0])
        self.clusteringGraph.startingPoints = []
        self.clusteringGraph.startingPoints.append(0) 
        
        self.clusteringGraph.CalculateMaxClustersTime()
        correctMaxClustersTime = np.array([3,1,2,4,1])
        
        self.assertTrue((correctMaxClustersTime == self.clusteringGraph.maxClustersTime).all()) 
        return
    
    def CheckCalculateMaxClustersTime_short_multipleStartingPoints(self):
        
        # Initialization
        self._InitializeDummyClusteringGraph()
        self.clusteringGraph.num_clusters = 5
        self.clusteringGraph.clustersSequence = \
           np.array([0,0,1,0,0   ,0,2,2,3,3,    3,3,4,0])
        self.clusteringGraph.startingPoints = []
        self.clusteringGraph.startingPoints.append(0) 
        self.clusteringGraph.startingPoints.append(5) 
        self.clusteringGraph.startingPoints.append(10) 
        
        self.clusteringGraph.CalculateMaxClustersTime()
        correctMaxClustersTime = np.array([2,1,2,2,1])
        
        self.assertTrue((correctMaxClustersTime == self.clusteringGraph.maxClustersTime).all()) 
        return
    
    def CheckCalculateMaxClustersTime_long(self):
        
        # Initialization
        self._InitializeDummyClusteringGraph()
        self.clusteringGraph.num_clusters = 5
        self.clusteringGraph.clustersSequence = \
           np.array([0,0,0,0,1,1,1,1,1,1,0,0,0,2,2,2,2,2,2,2,3,3,3,3,3,3,4,0,0,0,0,0])
        self.clusteringGraph.startingPoints = []
        self.clusteringGraph.startingPoints.append(0) 
        
        self.clusteringGraph.CalculateMaxClustersTime()
        correctMaxClustersTime = np.array([5,6,7,6,1])
        
        self.assertTrue((correctMaxClustersTime == self.clusteringGraph.maxClustersTime).all()) 
        return
    
    def CheckCalculateMaxClustersTime_long_multipleStartingPoints(self):
        
        # Initialization
        self._InitializeDummyClusteringGraph()
        self.clusteringGraph.num_clusters = 5
        self.clusteringGraph.clustersSequence = \
           np.array([0,0,0,0,1,1,1,1,1,1,0,0,0,2,2,2,2,2,2,2,3,3,3,3,3,3,4,0,0,0,0,0])
        self.clusteringGraph.startingPoints = []
        self.clusteringGraph.startingPoints.append(0) 
        self.clusteringGraph.startingPoints.append(6) 
        self.clusteringGraph.startingPoints.append(23) 
        
        self.clusteringGraph.CalculateMaxClustersTime()
        correctMaxClustersTime = np.array([5,4,7,3,1])
        
        self.assertTrue((correctMaxClustersTime == self.clusteringGraph.maxClustersTime).all()) 
        return
    
    def CheckEliminateOneCluster():
        
        return
    
    def CheckSmoothClusterAssignmentsAndRecalculateProperties():
        
        return
    
    @staticmethod
    def PerformAllTests():
        
        TestCG = TestsClusteringGraph()
        TestCG.CheckFindTransitionMatrix()
        TestCG.CheckFindTransitionMatrix_multipleStartingPoints()
        TestCG.CheckCalculateTemporalTransitionMatrices()
        TestCG.CheckCalculateTemporalTransitionMatrices_multipleStartingPoints()
        TestCG.CheckCalculateMaxClustersTime_short()
        TestCG.CheckCalculateMaxClustersTime_short_multipleStartingPoints()
        TestCG.CheckCalculateMaxClustersTime_long()
        TestCG.CheckCalculateMaxClustersTime_long_multipleStartingPoints()
        print('All tests have been successfully performed')
        return
    
def main():
    TestsClusteringGraph.PerformAllTests()
    
main() 

    
    
    
