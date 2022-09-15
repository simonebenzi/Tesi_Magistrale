

function [] = Alg210_ov_Final_VideoClusteringExtraction()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adding code paths and defining data paths
addpath('./MATLAB_paths');
paths = DefineCodeAndDataPaths();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Loading the outputs from KVAE and the Vocabularies
% Training GSs of odometry
[trainingGSs, ~] = loadObjectGivenFileName(paths.path_to_training_GSs);
% Odometry vocabulary
[odometryVocabulary, ~] = loadObjectGivenFileName(paths.path_to_cluster_vocabulary_odometry);
% Outputs from KVAE training
[a_states, ~] = loadObjectGivenFileName(paths.path_a_states_KVAE_training);
[z_states, ~] = loadObjectGivenFileName(paths.path_z_states_KVAE_training);
[alphas, ~] = loadObjectGivenFileName(paths.path_KVAE_alphas);
[newClustersSequence, ~] = loadObjectGivenFileName(paths.path_to_cut_cluster_sequence);
[vectorOfKeptOdometryClusters, ~] = loadObjectGivenFileName(paths.path_to_vector_of_kept_clusters);
[predictedParamsKVAE, ~] = loadObjectGivenFileName(paths.path_to_predicted_odometry_from_KVAE);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Modify cluster of odometry
odometryVocabularyNew = ModifyOdometryClusteringBasedOnCutClusters(odometryVocabulary, ...
    newClustersSequence, vectorOfKeptOdometryClusters);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Vocabulary extraction
% Find mean and covariance values over video, over the odometry clusters
videoVocabularyNew = PerformClusteringWithOdometryAssignments(a_states,z_states,alphas);
videoVocabularyNew.startingPoints = odometryVocabulary.startingPoints;
% Find covariances of odometry dynamics prediction vs. KVAE prediction of odometry
videoVocabularyNew = FindCovariancePredictionVsDMatrices(videoVocabularyNew, ...
    odometryVocabularyNew, trainingGSs, predictedParamsKVAE);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save
net = odometryVocabularyNew;
save(paths.path_to_cluster_vocabulary_odometry_new, 'net')
net = videoVocabularyNew;
save(paths.path_to_cluster_vocabulary_video_new, 'net')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Checking the vocabulary assignments of video w.r.t. odometry
clusterAssignments = CompareClusterAssignments(odometryVocabulary.N, ...
    odometryVocabularyNew.dataColorNode, a_states, videoVocabularyNew.nodesMean, ...
    trainingGSs);
%% Plotting the winning alpha across time for video states
lastValueToPrintForClusters = 2000;
PlotWinningAlphaAcrossTime(lastValueToPrintForClusters, ...
    odometryVocabularyNew, clusterAssignments)

end
