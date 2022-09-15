
function [] = ChoosingParameters(baseFolderPath, path_to_GSs, path_to_training_positions_min, ...
    path_to_training_positions_max, path_to_tests, path_to_grid_tests, path_to_names_test, ...
    path_predicted_params_direct, path_weights, path_predicted_params_combined, ...
    path_indices_swapping, path_to_config_KVAE_combined, prefixName)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adding code paths and defining data paths
addpath('./MATLAB_paths');
%% Positional data
% Load it
[validationGSs, ~] = loadObjectGivenFileName(path_to_GSs);
% ACTUAL max and min over x and y
[realMins, ~] = loadObjectGivenFileName(path_to_training_positions_min);
[realMaxs, ~] = loadObjectGivenFileName(path_to_training_positions_max);
% Denormalize the positional data
validationGSsDenorm = denormalizeSequence(validationGSs(:,1:2), realMins, realMaxs);
%% Extracting the values of parameters that were given
[parametersGrid, ~] = loadObjectGivenFileName(path_to_grid_tests);
[parametersGridNames, ~] = loadObjectGivenFileName(path_to_names_test);
% Where is the initial time instant saved in the grid?
for i = 1:size(parametersGridNames,1)
    if contains(parametersGridNames(i,:), 'initialTimeInstant')
        indexOfInitialTimeInstant = i;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Files of outputs
outputFolderBase = path_to_tests;
subFolders = dir(outputFolderBase);
% Number of subfolders
numberOfSubFolders = length(subFolders) - 2 - 3; % 2 are empty, 3 for saved files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Looping over the subfolders
distances_mean_video    = [];
distances_mean_odometry = [];
distances_mean_video_reweighted    = [];
distances_mean_odometry_reweighted = [];
distances_mean_video_resampling    = [];
distances_mean_odometry_resampling = [];
distances_mean_video_resampling_full = [];
distances_mean_odometry_resampling_full = [];
for subFolderIndex = 3 : numberOfSubFolders + 2

    subFolderIndex
    
    %% Select the initial time instant
    indexInGrid = subFolderIndex - 2;
    initialTimeInstant = parametersGrid(indexInGrid,indexOfInitialTimeInstant) + 1;
    validationGSsDenormCut = validationGSsDenorm(initialTimeInstant:end,:);
    
    %% Current subfolder
    currentSubfolder = subFolders(subFolderIndex).name;
    %% Values in subfolder
    predicted_params_direct = loadObjectGivenFileName(fullfile(outputFolderBase, ...
        currentSubfolder, path_predicted_params_direct));
    predicted_params_combined = loadObjectGivenFileName(fullfile(outputFolderBase, ...
        currentSubfolder, path_predicted_params_combined));
    particles_weights = loadObjectGivenFileName(fullfile(outputFolderBase, ...
        currentSubfolder, path_weights));
    indices_swapping = loadObjectGivenFileName(fullfile(outputFolderBase, ...
        currentSubfolder, path_indices_swapping));
    %% Length of data, and data taken for that length
    lengthOfTesting = min(size(predicted_params_combined, 1), ...
                          size(validationGSsDenormCut, 1));
    validationGSsDenormCut = validationGSsDenormCut(1:lengthOfTesting,:);
    predicted_params_combined = predicted_params_combined(1:lengthOfTesting,:,:);
    predicted_params_direct = predicted_params_direct(1:lengthOfTesting,:,:);
    indices_swapping = indices_swapping(1:lengthOfTesting,:);
    %% Denormalize the predicted odometry values
    predicted_params_direct = permute(predicted_params_direct, [1,3,2]);
    predicted_params_directDenorm = denormalizeArrayOfSequences(...
        predicted_params_direct(:,:,1:2), realMins, realMaxs);
    predicted_params_combined = permute(predicted_params_combined, [1,3,2]);
    predicted_params_combinedDenorm = denormalizeArrayOfSequences( ...
        predicted_params_combined(:,:,1:2), realMins, realMaxs);
    %% ----------------------------- DISTANCES ----------------------------
    %% Find distances of particles from real value for MJPF odometry
    [all_distances, ~, mean_distances] = ...
        FindParticleDistancesFromCurrentRealValue(validationGSsDenormCut, predicted_params_directDenorm);
    [all_distances_od, ~, mean_distances_od] = ...
        FindParticleDistancesFromCurrentRealValue(validationGSsDenormCut, predicted_params_combinedDenorm);
    % Adding mean errors (averaged across time)
    distances_mean_video    = [distances_mean_video; mean(mean_distances)];
    distances_mean_odometry = [distances_mean_odometry; mean(mean_distances_od)];
    %% Find distances, reweighted
    [~, mean_distances_reweighted] = WeightParticlesDistancesAndFindMean(all_distances, particles_weights);
    [~, mean_distances_reweighted_od] = WeightParticlesDistancesAndFindMean(all_distances_od, particles_weights);
    distances_mean_video_reweighted    = [distances_mean_video_reweighted, mean_distances_reweighted];
    distances_mean_odometry_reweighted = [distances_mean_odometry_reweighted, mean_distances_reweighted_od];
    
    %% Plotting
    %{
    figure
    scatter3(validationGSsDenorm(:,1), ...
        validationGSsDenorm(:,2), ...
        1:1:size(validationGSsDenorm,1), 'r')
    hold on
    for i = 1:size(predicted_params_combinedDenorm, 2)
        scatter3(predicted_params_combinedDenorm(:,i,1), ...
                predicted_params_combinedDenorm(:,i,2), ...
                1:1:size(validationGSsDenorm,1),'b')
    end
    %}
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Find the distances with corrections done by the resampling (video)
    realParamsDenorm = validationGSsDenormCut;
    predParams = predicted_params_directDenorm;
    newIndicesForSwapping = indices_swapping;
    [predParamsCorrected] = CutPredictionsBasedOnResampling(predParams, newIndicesForSwapping);
    [~, ~, mean_distances] = ...
        FindParticleDistancesFromCurrentRealValue(realParamsDenorm, predParamsCorrected);
    distances_mean_video_resampling = [distances_mean_video_resampling; mean(mean_distances)];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Find the distances with corrections done by the full resampling (video)
    realParamsDenorm = validationGSsDenormCut;
    predParams = predicted_params_directDenorm;
    newIndicesForSwapping = indices_swapping;
    [predParamsCorrected] = CutPredictionsBasedOnResamplingFull(predParams, newIndicesForSwapping);
    [~, ~, mean_distances] = ...
        FindParticleDistancesFromCurrentRealValue(realParamsDenorm, predParamsCorrected);
    distances_mean_video_resampling_full = [distances_mean_video_resampling_full; mean(mean_distances)];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Find the distances with corrections done by the resampling (odometry)
    realParamsDenorm = validationGSsDenormCut;
    predParams = predicted_params_combinedDenorm;
    newIndicesForSwapping = indices_swapping;
    [predParamsCorrected] = CutPredictionsBasedOnResampling(predParams, newIndicesForSwapping);
    [~, ~, mean_distances_od] = ...
        FindParticleDistancesFromCurrentRealValue(realParamsDenorm, predParamsCorrected);
    distances_mean_odometry_resampling = [distances_mean_odometry_resampling; mean(mean_distances_od)];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Find the distances with corrections done by the full resampling (odometry)
    realParamsDenorm = validationGSsDenormCut;
    predParams = predicted_params_combinedDenorm;
    newIndicesForSwapping = indices_swapping;
    [predParamsCorrected] = CutPredictionsBasedOnResamplingFull(predParams, newIndicesForSwapping);
    [~, ~, mean_distances_od] = ...
        FindParticleDistancesFromCurrentRealValue(realParamsDenorm, predParamsCorrected);
    distances_mean_odometry_resampling_full = [distances_mean_odometry_resampling_full; mean(mean_distances_od)];
end
%% Printing the results w.r.t. the grid of parameters
numberOfTestingParams = size(parametersGrid,2);
bullet_dimension = 10;
% parametersGridNames = char('$N$','$temp_v$', '$N_{th,h}$', ...
%     '$N_{th,l}$', '$R_{imp}$', '$t_{init}$');
offset_ratio = 0.1;
resolution = 1200;
% First plot with bars:
for i = 1:numberOfTestingParams
    
    figureDistances = figure
    figureDistances.Position = [200,200,300,450];
    
    currentParametersValues = parametersGrid(1:numberOfSubFolders,i);
    %currentParametersNames  = parametersGridNames(i,:);
    currentParametersNames  = parametersGridNames(i,:);
    % Means and stds
    uniqueParameters = unique(parametersGrid(1:numberOfSubFolders,i));
    %[means0, stds0] = FindMeanAndStdOfSubpartOfArray(distances_mean_video, ...
    %    currentParametersValues, uniqueParameters);
    %[means1, stds1] = FindMeanAndStdOfSubpartOfArray(distances_mean_odometry, ...
    %    currentParametersValues, uniqueParameters);
    [means2, stds2] = FindMeanAndStdOfSubpartOfArray(distances_mean_video_reweighted, ...
        currentParametersValues, uniqueParameters);
    [means3, stds3] = FindMeanAndStdOfSubpartOfArray(distances_mean_odometry_reweighted, ...
        currentParametersValues, uniqueParameters);
    [means4, stds4] = FindMeanAndStdOfSubpartOfArray(distances_mean_video_resampling_full, ...
        currentParametersValues, uniqueParameters);
    [means5, stds5] = FindMeanAndStdOfSubpartOfArray(distances_mean_odometry_resampling_full, ...
        currentParametersValues, uniqueParameters);
    
    x = 1:1:length(uniqueParameters);
    errorValuesBar = [];
    stdValuesBarHigh = [];
    stdValuesBarLow = [];
    for j = 1:length(uniqueParameters)
        %errorValue = [means0(j), means1(j), means2(j), means3(j), means4(j), means5(j)];
        errorValue = [means2(j), means3(j), means4(j), means5(j)];
        errorValuesBar = [errorValuesBar; errorValue];
        %stdHigh = [stds0(j), stds1(j), stds2(j), stds3(j), stds4(j), stds5(j)];
        stdHigh = [stds2(j), stds3(j), stds4(j), stds5(j)];
        stdLow  = min([stdHigh; errorValue], [], 1);
        stdValuesBarHigh = [stdValuesBarHigh; stdHigh];
        stdValuesBarLow = [stdValuesBarLow; stdLow];
    end
    b = bar(x,errorValuesBar)
    b(1).FaceColor = [0 0 1]; %blue
    b(2).FaceColor = [1 0 0]; %red
    b(3).FaceColor = [0 1 1]; %cyan
    b(4).FaceColor = [1 0 1]; %magenta
    hold on
    % From here:
    % https://it.mathworks.com/matlabcentral/answers/102220-how-do-i-place-errorbars-on-my-grouped-bar-graph-using-function-errorbar-in-matlab
    % Find the number of groups and the number of bars in each group
    ngroups = size(errorValuesBar,1);
    nbars = size(errorValuesBar,2);
    % Calculate the width for each bar group
    groupwidth = min(0.8, nbars/(nbars + 1.5));
    % Set the position of each error bar in the centre of the main bar
    % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
    for j = 1:nbars
        if i == 6
            capSize = 2;
        else
            capSize = 10;
        end
        % Calculate center of each bar
        barCenters = (1:ngroups) - groupwidth/2 + (2*j-1) * groupwidth / (2*nbars);
        errorbar(barCenters, errorValuesBar(:,j), stdValuesBarLow(:,j), stdValuesBarHigh(:,j), ...
            'k', 'linestyle', 'none', 'CapSize',capSize);
    end
    xlabel(currentParametersNames, 'FontSize',11,'Interpreter','latex')
    ylabel('Error (m)', 'FontSize',11,'Interpreter','latex')
    grid on
    xticks(x)
    xticklabels({uniqueParameters})
    %labels = uniqueParameters;
    
    % Save
    ax = gca;
    % Requires R2020a or later
    exportgraphics(ax, ...
        fullfile(baseFolderPath, ['bar_plot_', prefixName, '_', currentParametersNames, '.png']), ...
        'Resolution',resolution)
end
% Second plot with points:
for i = 1:numberOfTestingParams
    figureDistances = figure
    figureDistances.Position = [200,200,300,450];
    
    uniqueValues = unique(parametersGrid(1:numberOfSubFolders,i));
    minValue_x = uniqueValues(1);
    maxValue_x = uniqueValues(end);
    if length(uniqueValues)> 1
        offset_x = (maxValue_x-minValue_x)*offset_ratio;
    end
    %ones_vector = ones(numberOfSubFolders,1);
    %scatter(parametersGrid(1:numberOfSubFolders,i), ...
    %     distances_mean_video,bullet_dimension,'m', '*')
    hold on
    %scatter(parametersGrid(1:numberOfSubFolders,i), ...
    %     distances_mean_odometry,bullet_dimension,'c', '*')
    scatter(parametersGrid(1:numberOfSubFolders,i), ...
         distances_mean_video_reweighted,bullet_dimension,'b', '*')
    scatter(parametersGrid(1:numberOfSubFolders,i), ...
         distances_mean_odometry_reweighted,bullet_dimension,'r', '*')
    scatter(parametersGrid(1:numberOfSubFolders,i), ...
         distances_mean_video_resampling_full,bullet_dimension,'c', '*')
    scatter(parametersGrid(1:numberOfSubFolders,i), ...
         distances_mean_odometry_resampling_full,bullet_dimension,'m', '*')
    currentParametersNames = parametersGridNames(i,:);
    xlabel(currentParametersNames, 'FontSize',11, 'Interpreter','latex')
    ylabel('Error (m)', 'FontSize',11, 'Interpreter','latex')
    if length(uniqueValues)> 1
        xlim([minValue_x-offset_x  maxValue_x+offset_x])
    end
    grid on
    %legend({'mean video err.', 'mean odometry err.', ...
    %    'mean weighted video err.', 'mean weighted odometry err.', ...
    %    'mean video F.resam. err.', 'mean odom. F.resam. err.'},  ...
    %    'FontSize',9, 'Interpreter','latex')
    
    % Save
    ax = gca;
    % Requires R2020a or later
    exportgraphics(ax, ...
        fullfile(baseFolderPath, ['point_plot_', prefixName, '_', currentParametersNames, '.png']), ...
        'Resolution',resolution)
end
%}
%% Combining the estimation from the attempts with the same parameters
%  but different starting point.
differentStartingPoints = unique(parametersGrid(:,indexOfInitialTimeInstant));
howManyDifferentStartingPoints = length(differentStartingPoints);
howManyParameterRows = size(parametersGrid,1)/howManyDifferentStartingPoints;
% new grid
newParametersGrid = [];
% new parameters grid name (take out the starting point one)
newParametersGridNames = parametersGridNames;
newParametersGridNames(indexOfInitialTimeInstant,:) = [];
% Final errors
finalErrors = zeros(howManyParameterRows,1);
for i = 1:size(parametersGrid,1)
    currentOriginalRow = parametersGrid(i,:);
    currentOriginalRow(indexOfInitialTimeInstant) = [];
    if i ~= 1        
        for j = 1:size(newParametersGrid,1)
            isTheSameRow = isequal(currentOriginalRow,newParametersGrid(j,:));
            if isTheSameRow
                foundRow = j;
                break;
            end
        end
        if isTheSameRow == false
            newParametersGrid = [newParametersGrid; currentOriginalRow];
            foundRow = size(newParametersGrid,1); 
        end
    else
        newParametersGrid = [newParametersGrid; currentOriginalRow];
        foundRow = 1; 
    end    
    finalErrors(foundRow) = finalErrors(foundRow) + ...
        distances_mean_odometry_resampling_full(i);
end
%% Choosing the parameters that provided the lowest final error
[~,indexOfMinimumFinalError]  = min(finalErrors);
bestTestingParams = newParametersGrid(indexOfMinimumFinalError,:);
%% Saving the parameters in the configuration json
SaveConfigurationValuesToJSONFile(path_to_config_KVAE_combined, ...
    newParametersGridNames, bestTestingParams)
end