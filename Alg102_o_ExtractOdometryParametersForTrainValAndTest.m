
% Function to extract the parameters of motion, or the Generalized States.
% Check under Parameters_definition -> Configuration for parameters to be
% set for the filtering.

% To perform filtering over training, validation or testing, change the
% variable 'dataCase'.

function [] = Alg102_o_ExtractOdometryParametersForTrainValAndTest()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adding code paths and defining data paths
addpath('./MATLAB_paths');
paths = DefineCodeAndDataPaths();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Extract filtering parameters
paramsFiltering   = Config_filtering();
paramsClustering  = Config_clustering(); % just used for taking the weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXTRACTION
% 0 = training, 1 = validation, 2 = testing
for dataCase = 0:2
    dataCase
    % Name of input and output based on the data case
    if dataCase == 0
        path_to_positions_cells = paths.path_to_training_positions_cells_norm;
        path_to_GSs_folder      = paths.path_to_training_GSs_folder;
        path_to_GSs_cells       = paths.path_to_training_GSs_cells;
        path_to_GSs             = paths.path_to_training_GSs;
    elseif dataCase ==1
        path_to_positions_cells = paths.path_to_validation_positions_cells_norm;
        path_to_GSs_folder      = paths.path_to_validation_GSs_folder;
        path_to_GSs_cells       = paths.path_to_validation_GSs_cells;
        path_to_GSs             = paths.path_to_validation_GSs;
    elseif dataCase == 2
        path_to_positions_cells = paths.path_to_test_positions_cells_norm;
        path_to_GSs_folder      = paths.path_to_test_GSs_folder;
        path_to_GSs_cells       = paths.path_to_test_GSs_cells;
        path_to_GSs             = paths.path_to_test_GSs;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % LOADING the positions
    [data, isLoaded] = loadObjectGivenFileName(path_to_positions_cells);
    if isLoaded == true
        data   = data'; % Parameters holder wants the cells on first dimension
        
        %% Create an object of class DataHolder to store the data for training
        dataHolder         = DataHolder(data, data);
        % Number of data cells
        numCells  = size(dataHolder.Data,1);
        dataHolder.Plot2DDataSingleCellInTime(1);
        
        %% Perform Filtering
        %% Null Force Filter definition
        % Dimension of observation (e.g., positional data -> 2)
        observationDimension = size(dataHolder.Data{1,1}, 2)*2;
        % Create Null Force Filter
        NFF = NullForceFilterWithGivenVelocity(...
            paramsFiltering.obsVar, ...
            paramsFiltering.predVar, ...
            paramsFiltering.initialVar, ...
            observationDimension, ...
            observationDimension);
        % Perform Null Force Filtering            
        [estimatedStates] = NFF.PerformKalmanFiltering(dataHolder.DataPlusSin{1,1}');    
        KalmanFilter.Plot2DObservationsVsFiltering(dataHolder.DataPlusSin{1,1}', estimatedStates);
        %% Parameters extractor                           
        parametersHolder = ParametersHolder(dataHolder, NFF, ...
            paramsFiltering.memoryLength, paramsFiltering.memoryWeights);
        parametersHolder.PlotAllDataAndParameters2DFirstCell();
        
        %% Features of motion
        % 1) All together
        non_zero_features           = find(paramsClustering.weights ~= 0);
        features = parametersHolder.dataAndParametersInSingleArray(:, non_zero_features);
        % Plot all together
        figure
        scatter(features(:,1),features(:,2))
        figure
        plot(features(:,3))
        figure
        plot(features(:,4))
        % 2) Divided per cell
        features_per_cell = cell(size(data, 1), 1);
        for i = 1:size(data, 1)
            currentFeatures = parametersHolder.dataAndParameters{i,:}(:, non_zero_features);
            features_per_cell{i,1} = currentFeatures;
            %save([FeaturesFileName, '_', num2str(i)], 'features');
        end
        SaveDataToPathInFolder(path_to_GSs_folder, path_to_GSs_cells, features_per_cell);
        SaveDataToPathInFolder(path_to_GSs_folder, path_to_GSs, features);
    end
end

end
