
% This function aligns the latent states from the trained VAE with the 
% odometry data, for the training data.

% This means that a cell array is created for keeping the VAE latent
% states, and a single array for the odometry values.

function [] = Alg206_ov_AlignOutputOfVAEWithOdometry()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adding code paths and defining data paths
addpath('./MATLAB_paths');
paths = DefineCodeAndDataPaths();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DATA EXTRACTION
% Positional data
[dataPositions, isLoaded] = loadObjectGivenFileName(paths.path_to_training_positions_cells_norm);
if isLoaded == false
    throw(MException('MyComponent:noSuchVariable', 'Could not load training positional normalized data'))
end
% Image data
[dataImagesLatentStates, isLoaded] = loadObjectGivenFileName(paths.path_to_latent_state_from_VAE);
dataImagesLatentStates = squeeze(dataImagesLatentStates)';
if isLoaded == false
    throw(MException('MyComponent:noSuchVariable', 'Could not load training image VAE latent state data'))
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Alignment of video latent state data in cells
numberOfCells = size(dataPositions,2);
dataImagesLatentStatesCells = cell(1,numberOfCells);
beginOfCurrentCell = 1;
for i = 1:numberOfCells
    currentCellForPositionData = dataPositions{1,i};
    lengthOfCurrentCell = size(currentCellForPositionData,1);
    endOfCurrentCell = beginOfCurrentCell + lengthOfCurrentCell - 1;
    dataImagesLatentStatesCells{1,i} = dataImagesLatentStates(:,beginOfCurrentCell:endOfCurrentCell)';

    beginOfCurrentCell = beginOfCurrentCell + lengthOfCurrentCell;
end
% Check that we got to the end of the dataImagesLatentStates array
if endOfCurrentCell ~= size(dataImagesLatentStates, 2)
    throw(MException('MyComponent:outOfRange', 'Positional data and VAE video latent states have different length'))
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save aligned VAE images latent states
save(paths.path_to_latent_state_from_VAE_cells, 'dataImagesLatentStatesCells')

end