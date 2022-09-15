
% This function finds min and max of odometry over the training data and 
% normalizes it.
% Save the min and max values and use them to normalize the validation and 
% testing data too.

function [] = Alg101_o_NormalizeOdometryDataForTrainValAndTest()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adding code paths and defining data paths
addpath('./MATLAB_paths');
paths = DefineCodeAndDataPaths();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Normalizing
% 0 = training, 1 = validation, 2 = testing
for dataCase = 0:2
    % Name of input and output based on the data case
    if dataCase == 0
        path_to_positions_cells      = paths.path_to_training_positions_cells;
        path_to_positions_cells_norm = paths.path_to_training_positions_cells_norm;
    elseif dataCase ==1
        path_to_positions_cells      = paths.path_to_validation_positions_cells;
        path_to_positions_cells_norm = paths.path_to_validation_positions_cells_norm;
    elseif dataCase == 2
        path_to_positions_cells      = paths.path_to_test_positions_cells;
        path_to_positions_cells_norm = paths.path_to_test_positions_cells_norm;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Load and normalize
    [data, isLoaded] = loadObjectGivenFileName(path_to_positions_cells);
    if isLoaded == true
        data_norm = data;
        if dataCase == 0
            % If this is the training case, we have to find the max and min
            [dataMin, dataMax] = LookForMinAndMaxValuesInCellArray(data);
            % And we save it.
            save(paths.path_to_training_positions_min, 'dataMin')
            save(paths.path_to_training_positions_max, 'dataMax')
        else
            % If it is not the training, we have to load min and max values
            dataMin = loadObjectGivenFileName(paths.path_to_training_positions_min);
            dataMax = loadObjectGivenFileName(paths.path_to_training_positions_max);
        end
        % Now we normalize the data
        dataNorm = NormalizeAllValuesInCellArray(data, dataMin, dataMax);
        % And we save it
        save(path_to_positions_cells_norm, 'dataNorm')
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Now we plot
        figure
        hold on
        for i = 1:size(dataNorm, 2)
            scatter(dataNorm{1,i}(:,1), dataNorm{1,i}(:,2))
        end
    end
end

end
