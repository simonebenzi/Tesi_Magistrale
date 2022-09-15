
%% This function looks at the results obtained from the multiple 
% tests on the validation data performed in 
% 'Alg300_ov_TrackingValidationWithMultipleMJPFParameters' and chooses
% the combination of parameters that gave the lowest localization error.
% This combination of parameters is then saved in the 
% 'Config_KVAE_combined.json' file under 'ConfigurationFiles' so that it
% can be used in the next steps of the code.
% Plots are also produced and saved.

function [] = Alg301_ov_ChoosingParametersForTraining()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adding code paths and defining data paths
addpath('./MATLAB_paths');
paths = DefineCodeAndDataPaths();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Running the main code
baseFolderPath = paths.baseFolderPath;
path_to_GSs = paths.path_to_validation_GSs;
path_to_training_positions_min = paths.path_to_training_positions_min;
path_to_training_positions_max = paths.path_to_training_positions_max;
path_to_tests = paths.path_to_first_tests_on_validation;
path_to_grid_tests = paths.path_to_grid_first_tests_on_validation;
path_to_names_test = paths.path_to_names_first_tests_on_validation;
path_predicted_params_direct = paths.predicted_params_direct_file_name;
path_weights = paths.path_weights;
path_predicted_params_combined = paths.predicted_params_combined_file_name;
path_indices_swapping = paths.indices_swapping_file_name;
path_to_config_KVAE_combined = paths.path_to_config_KVAE_combined;
prefixName = 'Params';
ChoosingParameters(baseFolderPath,path_to_GSs, path_to_training_positions_min, ...
    path_to_training_positions_max, path_to_tests ,path_to_grid_tests, ...
    path_to_names_test, path_predicted_params_direct, path_weights, ...
    path_predicted_params_combined, ...
    path_indices_swapping, path_to_config_KVAE_combined, prefixName)

end