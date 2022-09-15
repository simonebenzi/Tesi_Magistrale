
function [data, testingLength, alphas_from_assignments_video, ...
    alphas_from_assignments_odometry, realParamsDenorm, ...
    videoPredOdometryDenorm, odometryUpdatedOdometryDenorm] = ...
    ExtractTrackingDebugInformation(paths,dataCase, endingPoint)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ACTUAL max and min over x and y
[realMins, ~] = loadObjectGivenFileName(paths.path_to_training_positions_min);
[realMaxs, ~] = loadObjectGivenFileName(paths.path_to_training_positions_max);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Selection of odometry file, of dictionary file, alpha, predicted params
% and mus file.
files = DefineAllCombinedMJPFsRelatedFilesDebugCode110();
if dataCase == 0 % train (0), validation (1) or test (2)
    files.realOdometryFile = paths.path_to_training_GSs;
elseif dataCase == 1
    files.realOdometryFile = paths.path_to_validation_GSs;
elseif dataCase == 2
    files.realOdometryFile = paths.path_to_test_GSs;
end
files.dictionaryFile      = paths.path_to_cluster_vocabulary_odometry_new;
files.alphaFile           = paths.path_KVAE_alphas;
files.predictedParamsFile = paths.path_to_predicted_odometry_from_KVAE;
files.musFile             = paths.path_a_states_KVAE_training;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Loading the files
data = LoadAllFilesDebugCode110(files, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Defining lengths
if endingPoint ~= -1
    testingLength = min(endingPoint, size(data.odometryUpdatedParticlesTestOd, 1)) - 1;
else
    testingLength = size(data.odometryUpdatedParticlesTestOd, 1);
end
testingLength = min(testingLength, size(data.data_sync,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Number of Clusters
numberOfClusters = size(data.net.nodesMean,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Find alphas from cluster assignments
alphas_from_assignments_video    = FindAlphasFromClusterAssignments( ...
    numberOfClusters, data.clusterAssignmentsTest);
alphas_from_assignments_odometry = FindAlphasFromClusterAssignments( ...
    numberOfClusters, data.clusterAssignmentsTestOd);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Denormalize the real and predicted odometry values
% For real params
realParamsDenorm           = denormalizeSequence(...
    data.data_sync(:,1:2), realMins, realMaxs);
% For particles video
videoPredOdometry          = permute(data.predictedParams_test, [1,3,2]);
videoPredOdometryDenorm    = denormalizeArrayOfSequences(...
    videoPredOdometry(:,:,1:2), realMins, realMaxs);
% For predicted all
odometryUpdatedOdometryDenorm = denormalizeArrayOfSequences( ...
    data.odometryUpdatedParticlesTestOd(:,:,1:2), realMins, realMaxs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end