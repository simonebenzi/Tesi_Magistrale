
function [] = Alg100_ov_Divide_in_train_and_validation()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adding code paths and defining data paths
addpath('./MATLAB_paths');
paths = DefineCodeAndDataPaths();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters
paramsTraining = Config_data_partitioning();
percentageOfTrajectoriesForTraining = paramsTraining.percentageOfTrajectoriesForTraining;

%% POSITIONAL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load the positional data
[positions, isLoaded] = loadObjectGivenFileName(paths.path_to_positions_cells);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Divide the positional data in training and validation
totalNumberOfTrajectories      = size(positions,2);
numberOfTrainingTrajectories   = ceil(percentageOfTrajectoriesForTraining*totalNumberOfTrajectories);
numberOfValidationTrajectories = totalNumberOfTrajectories - numberOfTrainingTrajectories;
% Divide
trainPositions      = cell(1, numberOfTrainingTrajectories);
validationPositions = cell(1, numberOfValidationTrajectories);
for i = 1:numberOfTrainingTrajectories
    trainPositions{1,i} = positions{1,i};
end
for i = 1:numberOfValidationTrajectories
    validationPositions{1,i} = positions{1,numberOfTrainingTrajectories+i};
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the positional data
% Training
% Empty the folder first
delete(paths.path_to_training_positions_cells)
% Save
SaveDataToPathInFolder(paths.path_to_training_positions_folder, ...
    paths.path_to_training_positions_cells, trainPositions)
% Validation
% Empty the folder first
delete(paths.path_to_validation_positions_folder)
% Save
SaveDataToPathInFolder(paths.path_to_validation_positions_folder, ...
    paths.path_to_validation_positions_cells, validationPositions)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% IMAGE DATA
names_of_image_data_folders = dir(paths.path_to_images_folder);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check if there is some cell/folder with dimension that is different
for i = 1:size(positions,2)
    number_of_elements_position = size(positions{1,i},1);
    iFolderName = names_of_image_data_folders(i+2).name;
    namesOfImagesInFolder_i = dir(fullfile(paths.path_to_images_folder, iFolderName));
    number_of_elements_image = size(namesOfImagesInFolder_i,1)-2;
    if number_of_elements_image ~= number_of_elements_position
        throw(MException('MyComponent:outOfRange', ...
            ['Positional data and VAE video latent states have different length: ', ...
            'folder', iFolderName]));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Empty the folders first (if there were previous elements remaining, it
% would be a big problem later)
CheckWhetherFolderExistsAndDeleteIt(paths.path_to_training_images_folder)
CheckWhetherFolderExistsAndDeleteIt(paths.path_to_validation_images_folder)
% Then create them again
CheckWhetherFolderExistsOrCreateIt(paths.path_to_training_images_folder);
CheckWhetherFolderExistsOrCreateIt(paths.path_to_validation_images_folder);
% Train 
for i = 1 : numberOfTrainingTrajectories
    i
    iFolderName = names_of_image_data_folders(i+2).name;
    inputPath   = fullfile(paths.path_to_images_folder, iFolderName);
    outputPath  = fullfile(paths.path_to_training_images_folder, iFolderName);
    copyfile(inputPath, outputPath)
end
% Validation
for i = numberOfTrainingTrajectories+1 : totalNumberOfTrajectories
    i
    iFolderName = names_of_image_data_folders(i+2).name;
    inputPath   = fullfile(paths.path_to_images_folder, iFolderName);
    outputPath  = fullfile(paths.path_to_validation_images_folder, iFolderName);
    copyfile(inputPath, outputPath)
end

end
