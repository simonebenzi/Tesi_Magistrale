
% This function takes the base path to where all the Carla data is stored
% and extracts the paths to every piece of data. The paths are saved in the
% variable 'paths'.
function [paths] = DefinePathsToData(baseFolderPath)

%% Define the paths
paths.baseFolderPath                            = baseFolderPath;
% Paths to the output from Carla
paths.path_to_positions_folder                  = fullfile(baseFolderPath, 'position');
paths.path_to_images_folder                     = fullfile(baseFolderPath, 'camera');
paths.path_to_positions_cells                   = fullfile(paths.path_to_positions_folder, 'train_positions.mat');
paths.path_to_test_positions_folder             = fullfile(baseFolderPath, 'test_position');
paths.path_to_images_folder_test                = fullfile(baseFolderPath, 'test_images');
paths.path_to_test_positions_cells              = fullfile(paths.path_to_test_positions_folder, 'test_positions.mat');
% Paths to the division in training and validation of the original Carla
% data
paths.path_to_training_positions_folder         = fullfile(baseFolderPath, 'train_positions');
paths.path_to_training_positions_cells          = fullfile(paths.path_to_training_positions_folder, 'train_positions.mat');
paths.path_to_training_positions_min            = fullfile(paths.path_to_training_positions_folder, 'train_positions_min.mat');
paths.path_to_training_positions_max            = fullfile(paths.path_to_training_positions_folder, 'train_positions_max.mat');
paths.path_to_training_positions_cells_norm     = fullfile(paths.path_to_training_positions_folder, 'train_positions_norm.mat');
paths.path_to_training_images_folder            = fullfile(baseFolderPath, 'train_images');
paths.path_to_validation_positions_folder       = fullfile(baseFolderPath, 'validation_positions');
paths.path_to_validation_positions_cells        = fullfile(paths.path_to_validation_positions_folder, 'validation_positions.mat');
paths.path_to_validation_positions_cells_norm   = fullfile(paths.path_to_validation_positions_folder, 'validation_positions_norm.mat');
paths.path_to_validation_images_folder          = fullfile(baseFolderPath, 'validation_images');
paths.path_to_test_positions_cells_norm         = fullfile(paths.path_to_test_positions_folder, 'test_positions_norm.mat');
paths.path_to_test_images_folder                = fullfile(baseFolderPath, 'test_images');
% Paths to the Generalized Filters for training and validation
paths.path_to_training_GSs_folder               = fullfile(baseFolderPath, 'train_GSs');
paths.path_to_training_GSs                      = fullfile(paths.path_to_training_GSs_folder, 'train_GSs.mat');
paths.path_to_training_GSs_cells                = fullfile(paths.path_to_training_GSs_folder, 'train_GSs_cells.mat');
paths.path_to_validation_GSs_folder             = fullfile(baseFolderPath, 'validation_GSs');
paths.path_to_validation_GSs                    = fullfile(paths.path_to_validation_GSs_folder, 'validation_GSs.mat');
paths.path_to_validation_GSs_cells              = fullfile(paths.path_to_validation_GSs_folder, 'validation_GSs_cells.mat');
paths.path_to_test_GSs_folder                   = fullfile(baseFolderPath, 'test_GSs');
paths.path_to_test_GSs                          = fullfile(paths.path_to_test_GSs_folder, 'test_GSs.mat');
paths.path_to_test_GSs_cells                    = fullfile(paths.path_to_test_GSs_folder, 'test_GSs_cells.mat');
% Path to latent state extracted from VAE
paths.path_to_python_outputs                    = fullfile(baseFolderPath, 'trained_VAE');
paths.path_to_latent_state_from_VAE             = fullfile(paths.path_to_python_outputs, 'TRAIN_a_states.mat');
paths.path_to_latent_state_from_VAE_cells       = fullfile(paths.path_to_python_outputs, 'TRAIN_a_states_cells.mat');
% Path to saved odometry plot
paths.path_saved_odometry_plot                  = fullfile(baseFolderPath, 'Odometry_plot');
% Path to saved video of images vs. odometry
paths.path_to_video_of_images_vs_odometry       = fullfile(baseFolderPath, 'Images_vs_odometry');
% Path to saved vocabulary
paths.path_to_cluster_vocabulary_odometry       = fullfile(baseFolderPath, 'Vocabulary_odometry.mat');
paths.path_to_cluster_vocabulary_video          = fullfile(baseFolderPath, 'Vocabulary_video.mat');
paths.path_to_cluster_vocabulary_full_structure = fullfile(baseFolderPath, 'Vocabulary_full_structure.mat');
paths.path_to_clustering_plot                   = fullfile(baseFolderPath, 'Clustering_plot');
% Path to outputs from KVAE training
paths.path_to_python_outputs_KVAE               = fullfile(baseFolderPath, 'trained_KVAE');
paths.path_a_states_KVAE_training               = fullfile(paths.path_to_python_outputs_KVAE, 'TRAIN_single_epoch_a_states.mat');
paths.path_z_states_KVAE_training               = fullfile(paths.path_to_python_outputs_KVAE, 'TRAIN_single_epoch_z_states.mat');
paths.path_KVAE_alphas                          = fullfile(paths.path_to_python_outputs_KVAE, 'TRAIN_single_epoch_alphas.mat');
paths.path_A_matrices_train                     = fullfile(paths.path_to_python_outputs_KVAE, 'TRAIN_single_epoch_As.mat');
paths.path_B_matrices_train                     = fullfile(paths.path_to_python_outputs_KVAE, 'TRAIN_single_epoch_Bs.mat');
paths.path_to_cut_cluster_sequence              = fullfile(paths.path_to_python_outputs_KVAE, 'newClustersSequence.mat');
paths.path_to_vector_of_kept_clusters           = fullfile(paths.path_to_python_outputs_KVAE, 'vectorOfKeptOdometryClusters.mat');
paths.path_to_predicted_odometry_from_KVAE      = fullfile(paths.path_to_python_outputs_KVAE, 'TRAIN_single_epoch_predicted_params_min.mat');
paths.path_to_cluster_vocabulary_odometry_new   = fullfile(baseFolderPath, 'Vocabulary_odometry_cut.mat');
paths.path_to_cluster_vocabulary_video_new      = fullfile(baseFolderPath, 'Odometry_based_vocabulary.mat');
paths.path_to_vocab_memory_verification         = fullfile(baseFolderPath, 'Vocabulary_for_memory_verification.mat');
paths.path_to_vocab_memory_verification_smaller = fullfile(baseFolderPath, 'Vocabulary_for_memory_verification_smaller.mat');
% Path to output from multiple parameters testing on validation data
paths.path_to_first_tests_on_validation         = fullfile(baseFolderPath, 'Validation_multiple_params');
paths.path_to_grid_first_tests_on_validation    = fullfile(paths.path_to_first_tests_on_validation, 'parametersGrid.mat');
paths.path_to_names_first_tests_on_validation   = fullfile(paths.path_to_first_tests_on_validation, 'parametersGrid_names.mat');
paths.predicted_params_direct_file_name         = 'OD_predictedParams_debugCode110.mat';
paths.path_weights                              = 'OD_particlesWeights_debugCode110.mat';
paths.predicted_params_combined_file_name       = 'OD_odometryUpdatedParticles_od_debugCode110.mat';
paths.indices_swapping_file_name                = 'OD_newIndicesForSwapping_numpy_debugCode110.mat';
% Path to combined MJPF configuration
paths.path_to_configuration_files               = fullfile('./ConfigurationFiles');
paths.path_to_config_KVAE_combined              = fullfile(paths.path_to_configuration_files, 'Config_KVAE_combined.json');
% Path to outputted anomalies from testing on training dataset
paths.path_to_test_on_training                  = fullfile(baseFolderPath, 'Tracking_results_training_best_parameters');
paths.path_to_training_anomalies                = fullfile(paths.path_to_test_on_training, 'OD_anomalies_debugCode110.mat');
% Path to output from multiple parameters testing on validation data
paths.path_to_second_tests_on_validation        = fullfile(baseFolderPath, 'Validation_multiple_threshs');
paths.path_to_grid_second_tests_on_validation   = fullfile(paths.path_to_second_tests_on_validation, 'parametersGrid.mat');
paths.path_to_names_second_tests_on_validation  = fullfile(paths.path_to_second_tests_on_validation, 'parametersGrid_names.mat');
% Path to output of test dataset tracking results
paths.path_to_tracking_output                   = fullfile(baseFolderPath, 'Tracking_results');
% Path for on-the-fly-testing-data
paths.path_to_on_the_fly_images_base_folder     = fullfile(baseFolderPath, 'On_the_fly_test_images');
paths.path_to_on_the_fly_images_final_folder    = fullfile(paths.path_to_on_the_fly_images_base_folder, 'test_images');
paths.path_to_on_the_fly_positions_folder       = fullfile(baseFolderPath, 'On_the_fly_test_positions');
paths.path_to_on_the_fly_positions_file         = fullfile(paths.path_to_on_the_fly_positions_folder, 'test_positions.mat');
paths.path_to_on_the_fly_reconstructed_data     = fullfile(baseFolderPath, 'On_the_fly_reconstructed_images');
% Path to video outputs
paths.path_to_video_outputs_folder              = fullfile(baseFolderPath, 'Video_outputs');
paths.video_output_name                         = 'real_time_video.avi';

return



