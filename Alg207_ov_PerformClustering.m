
% This function builds a clustering over positional data using the
% parameters of motion.
% In particular, the parameters to be considered should be defined in the
% file 
%  ----> ConfigurationFiles/Config_clustering <-----
% in the configuration file

function [] = Alg207_ov_PerformClustering()


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adding code paths and defining data paths
addpath('./MATLAB_paths');
paths = DefineCodeAndDataPaths();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Configuration parameters
% General parameters
paramsFiltering = Config_filtering();
% Parameters specific to clustering
paramsClustering = Config_clustering();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DATA EXTRACTION
% Training GSs of odometry
[trainingGSs, ~] = loadObjectGivenFileName(paths.path_to_training_GSs);
% Positional data
[dataPositions, isLoaded] = loadObjectGivenFileName(paths.path_to_training_positions_cells_norm);
if isLoaded == false
    throw(MException('MyComponent:noSuchVariable', 'Could not load training positional normalized data'))
end
dataPositions = dataPositions';
% Image data
[dataImagesLatentStates, isLoaded] = loadObjectGivenFileName(paths.path_to_latent_state_from_VAE);
if isLoaded == false
    throw(MException('MyComponent:noSuchVariable', 'Could not load training image VAE latent state data'))
end
dataImagesLatentStates = squeeze(dataImagesLatentStates);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create an object of class DataHolder to store the data for training
dataHolderPositions = DataHolder(dataPositions, dataPositions);
% Number of data cells
numCells   = size(dataHolderPositions.Data,1);
dataHolderPositions.Plot2DDataSingleCellInTime(1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Perform Filtering
%% Null Force Filter definition
% Dimension of observation (e.g., positional data -> 2)
observationDimension = size(dataHolderPositions.Data{1,1}, 2)*2;
% Create Null Force Filter
NFF = NullForceFilterWithGivenVelocity(...
    paramsFiltering.obsVar, ...
    paramsFiltering.predVar, ...
    paramsFiltering.initialVar, ...
    observationDimension, ...
    observationDimension);
% Perform Null Force Filtering            
[estimatedStates] = NFF.PerformKalmanFiltering(dataHolderPositions.DataPlusSin{1,1}');    
KalmanFilter.Plot2DObservationsVsFiltering(dataHolderPositions.DataPlusSin{1,1}', estimatedStates);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters extractor                           
parametersHolder = ParametersHolder(dataHolderPositions, NFF, ...
    paramsFiltering.memoryLength, paramsFiltering.memoryWeights);
parametersHolder.PlotAllDataAndParameters2DFirstCell();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Combine positional data and image VAE latent states
combinedData = [parametersHolder.dataAndParametersInSingleArray dataImagesLatentStates];
combinedData(isnan(combinedData)) = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Using VAE latent states?
a_dim = size(dataImagesLatentStates,2);
numberOfUsedPositionalFeatures = sum(paramsClustering.weights);
weightsOdometry = paramsClustering.weights;
if paramsClustering.considerVAEStates == true
    weightsVAELatentStates = ones(1,a_dim)*numberOfUsedPositionalFeatures/a_dim;
    paramsClustering.weights = [paramsClustering.weights,weightsVAELatentStates];
    inputData = combinedData;
else
    inputData = parametersHolder.dataAndParametersInSingleArray;
end
%% Using orthogonal GNG?
if paramsClustering.orthogonalGNG == true
    weightsForNetwork = paramsClustering.weightsForNetworkOrthogonal;
else
    weightsForNetwork = paramsClustering.weights;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clustering
%% MAIN GRAPH
if paramsClustering.orthogonalGNG == true
    clusteringGraph = ClusteringGraphWithParametersAndRestrictionOnOrthogonal;
    clusteringGraph = ...
    clusteringGraph.PerformGNGClustering(inputData, ...
                                         dataHolderPositions, parametersHolder, ...
                                         paramsClustering, weightsForNetwork); 
else
    clusteringGraph = ClusteringGraph;
    clusteringGraph = ...
    clusteringGraph.PerformGNGClustering(inputData, dataHolderPositions, ...
                                         paramsClustering, weightsForNetwork); 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Saving the Graph and the features
numberOfOdometryNonZeroFeatures = sum(weightsOdometry ~= 0);
nonZeroFeatures = find(paramsClustering.weights ~= 0);
nonZeroFeaturesOdometry = nonZeroFeatures(1:numberOfOdometryNonZeroFeatures); 
nonZeroFeaturesVideo = size(inputData,2)+1:size(combinedData,2); 
%% Odometry graph
% Only take necessary cluster features, and bring out of object, as 
% python will not read it otherwise
[clusteringSubGraphOdometry] = ExtractSubGraphInSelectionOfFeatures(clusteringGraph, ...
    combinedData, nonZeroFeaturesOdometry);
%% Video Graph
[clusteringSubGraphVideo]    = ExtractSubGraphWithDirectCalculation(clusteringGraph, ...
    combinedData, nonZeroFeaturesVideo);
%% Save vocabularies
net = clusteringSubGraphOdometry;
save(paths.path_to_cluster_vocabulary_odometry, 'net');
net = clusteringSubGraphVideo;
save(paths.path_to_cluster_vocabulary_video, 'net');
save(paths.path_to_cluster_vocabulary_full_structure , 'clusteringGraph');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plotting and save all the clusters, also with their std.
figure
hold on
PlotVocabularyErrorEllipses(clusteringSubGraphOdometry)
PlotNumberedClustersPosition(clusteringSubGraphOdometry)
% Save
ax = gca;
% Requires R2020a or later
resolution = 1200;
exportgraphics(ax, [paths.path_to_clustering_plot '.png'],'Resolution',resolution) 
%% Checking the vocabulary assignments of video w.r.t. odometry
clusterAssignments = CompareClusterAssignments(clusteringGraph.N, ...
    clusteringGraph.clusterAssignments, dataImagesLatentStates, ...
    clusteringSubGraphVideo.nodesMean, trainingGSs);
%% Plotting the winning alpha across time for video states
lastValueToPrintForClusters = 2000;
PlotWinningAlphaAcrossTime(lastValueToPrintForClusters, ...
    clusteringGraph, clusterAssignments)
end
