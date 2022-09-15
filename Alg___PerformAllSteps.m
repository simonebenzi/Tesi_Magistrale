
clear
clc
close all

%% User settings
% Were training and validation already separated or do you want the code
% to split the two based on the percentage you inserted in 
% 'Config_data_partitioning'?
separateTrainingAndValidation = false;
% Do you wish to make a video out of the training trajectories, showing
% images and corresponding odometry values time instant by time instant?
videoMaking = false;
% If video is performed, define frame rate and maximum number of
% trajectories to show
frame_rate = 10;
max_number_of_trajectories = 2;
%% Adding code paths and defining data paths
addpath('./MATLAB_paths');
paths = DefineCodeAndDataPaths();
%% Retrieve if you wish to use anomaly thresholds (i.e., particles restarting)
% Read the configuration file
jsonText = fileread(paths.path_to_config_KVAE_combined);
jsonData = jsondecode(jsonText);
usingAnomalyThresholds = getfield(jsonData,'usingAnomalyThresholds');

%% For python environment
getenv('PATH');

%% Pre-elaboration of the dataset (codes 1)
if separateTrainingAndValidation == true
    disp('Diving the data between training and validation')
    Alg100_ov_Divide_in_train_and_validation()
end
disp('Normalizing the data of training, validation and testing')
Alg101_o_NormalizeOdometryDataForTrainValAndTest()
disp('Extract the odometry parameters of training, validation and testing')
Alg102_o_ExtractOdometryParametersForTrainValAndTest()
disp('Plot odometries of training, validation and testing data')
Alg103_o_PlotTrajectories()
if videoMaking == true
    disp('Make a video of the training trajectories')
    Alg104_ov_MakeVideoOfTrajectory(frame_rate, max_number_of_trajectories);
end

%% Training of the models (codes 2)
disp('Training the Variational Autoencoder')
system('py Alg205_v_TrainVAE.py');
disp('Aligning latent state of VAE and odometry')
Alg206_ov_AlignOutputOfVAEWithOdometry()
disp('Perform clustering')
Alg207_ov_PerformClustering()
disp('Extract data for KVAE training')
system('py Alg208_ov_ExtractDataFileWithVideoAndParametersSingleTrajectory.py');
disp('Training the Kalman Variational Autoencoder')
system('py Alg209_ov_TrainKVAE.py');
disp('Final video clustering extraction')
Alg210_ov_Final_VideoClusteringExtraction();

%% Testing (codes 3)
% First multiple tests on validation for choosing the best parameters
disp('Testing on validation with multiple parameters')
system('py Alg300_ov_TrackingValidationWithMultipleMJPFParameters.py');
%
disp('Find best parameters')
Alg301_ov_ChoosingParametersForTraining()
if usingAnomalyThresholds == true
    % Then find anomaly mean and std w.r.t. training
    disp('Testing on training with best parameters')
    system('py Alg302_ov_TrackingTrainingWithBestParameters.py');
    %
    disp('Find mean and standard deviation of training anomalies')
    Alg303_FindMeanAndStdOfTrainingAnomalies()
    % Retest on validation for best thresholds
    disp('Testing on validation with multiple thresholds')
    system('py Alg304_ov_TrackingValidationWithMultipleThresholds.py');
    %
    disp('Find best thresholds parameters')
    Alg305_ov_ChoosingParametersForTesting()
end
% Finally test on testing data
disp('Test on testing data')
system('py Alg305_ov_TrackingOnTesting.py');
%
disp('Print tracking results on testing data')
Alg306_ov_PrintTrackingResults()


