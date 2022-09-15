function [data] = LoadAllFilesDebugCode110(files, first_time_instant)

addpath('../../General_utils_code/')

[data.net, ~]                    = loadObjectGivenFileName(files.dictionaryFile);

[data.mus_train, ~]              = loadObjectGivenFileName(files.musFile);

[data_sync, loaded]              = loadObjectGivenFileName(files.realOdometryFile);
if loaded == true
    if size(data_sync,1) < size(data_sync, 2)
        data_sync = data_sync';
    end
    data.data_sync               = data_sync(first_time_instant:end,:);
end

[data.alpha_train, ~]            = loadObjectGivenFileName(files.alphaFile);

[data.predictedParams_train, ~]  = loadObjectGivenFileName(files.predictedParamsFile);

[data.alpha_test, loaded]        = loadObjectGivenFileName(files.alphaTestFile);
if loaded == true
    data.alpha_test              = squeeze(data.alpha_test);
end

[data.predictedParams_test, loaded]   = loadObjectGivenFileName(files.predictedParamsTestFile);
if loaded == true
    data.predictedParams_test         = squeeze(squeeze(data.predictedParams_test));
    data.first_odometry_from_video    = data.predictedParams_test(1,:,:);
    data.second_odometry_from_video   = data.predictedParams_test(2,:,:);
    if size(data.predictedParams_test, 1) >= 1000
        data.thousand_odometry_from_video   = data.predictedParams_test(1000,:,:);
    else
        data.thousand_odometry_from_video = [];
    end
end

[data.mus_test, ~]               = loadObjectGivenFileName(files.musTestFile);

[data.clusterAssignmentsTest, loaded] = loadObjectGivenFileName(files.clusterAssignmentFile);
if loaded == true
    data.clusterAssignmentsTest  = squeeze(data.clusterAssignmentsTest);
end

[data.mean_videoMuAnomalies, loaded] = loadObjectGivenFileName(files.abn2File);
if loaded == true
    data.mean_videoMuAnomalies       = mean(data.mean_videoMuAnomalies, 2);
end

[passagesBetweenClustersTest, loaded] = loadObjectGivenFileName(files.clusterPassages);
if loaded == true
    data.passagesBetweenClustersTest  = permute(passagesBetweenClustersTest, [2,3,1]);
end

[data.timeInClustersTest, ~]     = loadObjectGivenFileName(files.timeInClustersFile);

data.particlesWeights         = loadObjectGivenFileName(files.particlesWeightsFile);

%% Odometric part

%data.predictedParams_test_od  = squeeze(squeeze( ...
%    loadObjectGivenFileName(files.predictedParamsTestOdFile)));

[data.clusterAssignmentsTestOd, loaded] = loadObjectGivenFileName(files.clusterAssignmentOdFile);
if loaded == true
    data.clusterAssignmentsTestOd = squeeze(data.clusterAssignmentsTestOd);
end

[odometryUpdatedParticles, ~]      = loadObjectGivenFileName(files.updatedOdometriesOd);
if loaded == true
    data.odometryUpdatedParticlesTestOd   = permute(odometryUpdatedParticles, [1,3,2]);
end

[odometryPredictedParticles, ~]    = loadObjectGivenFileName(files.predictedOdometriesOd);
if loaded == true
    data.odometryPredictedParticlesTestOd = permute(odometryPredictedParticles, [1,3,2]);
end

[data.timeInClustersTestOd, ~]     = loadObjectGivenFileName(files.timeInClustersOdFile);

%data.probs                    = loadObjectGivenFileName(files.probsReweightOdometries);

%data.indicesCrossAssignments  = loadObjectGivenFileName(files.crossAssignmentsFile);

%data.meanValuesAssigned       = loadObjectGivenFileName(files.meanValuesAssignedFile);

%data.updatedValuesOdometryBeforeResampling = ...
%    loadObjectGivenFileName(files.updatedValuesOdometryBeforeResampling);

%data.first_odometry_from_odometry = data.updatedValuesOdometryBeforeResampling(1,:,:);

%data.first_crossAssignments       = data.indicesCrossAssignments(1,:,:);
%data.first_meanValuesAssigned     = data.meanValuesAssigned(1,:,:);

%data.mjpf_clustersToPick                  = ...
%    loadObjectGivenFileName(files.mjpf_clustersToPickInResamplingFile);
%data.mjpf_clusterProbabilities            = ...
%    loadObjectGivenFileName(files.mjpf_clusterProbabilitiesInResamplingFile);
%data.mjpf_clusterProbabilities_new        = ...
%    loadObjectGivenFileName(files.mjpf_clusterProbabilities_newInResamplingFile);
%data.mjpf_indicesParticlesPerCluster      = ...
%    loadObjectGivenFileName(files.mjpf_indicesParticlesPerClusterInResamplingFile);

[data.very_first_mjpf_video_assignments, ~]    = ...
    loadObjectGivenFileName(files.very_first_mjpf_video_assignmentsFile);
[data.very_first_mjpf_odometry_assignments, ~] = ...
    loadObjectGivenFileName(files.very_first_mjpf_odometry_assignmentsFile);

[data.newIndicesForSwapping, ~]                = ...
    loadObjectGivenFileName(files.newIndicesForSwappingFile);
%data.newIndicesForSwappingReshaped        = ...
%    loadObjectGivenFileName(files.newIndicesForSwappingReshapedFile);

%data.clusterAssignmentsTrain = loadObjectGivenFileName(files.clusterAssignmentsTrainFile);

[data.whenResampled, ~]                = ...
    loadObjectGivenFileName(files.whenResampledFile);

%% Files related to updated and predicted and real a states
[data.aStatesPredictions, ~]    = loadObjectGivenFileName(files.aStatesPredictionsFile);
[data.aStatesUpdates, ~]        = loadObjectGivenFileName(files.aStatesUpdatesFile);
[data.aStatesReal, ~]           = loadObjectGivenFileName(files.aStatesRealFile);

%% Files related to the clustering
% Transition matrix between clusters
[data.transitionMat, ~]         = loadObjectGivenFileName(files.transitionMatFile);
% Temporal transition matrices windowed over times
[windowedtransMatsTime, ~]      = loadObjectGivenFileName(files.windowedtransMatsTimeFile);
if loaded == true
data.windowedtransMatsTime      = permute(windowedtransMatsTime, [2,3,1]);
end
% Max time spent in each cluster
[data.maxClustersTime, ~]       = loadObjectGivenFileName(files.maxClustersTimeFile);
% Matrix with nodes distances
[data.nodesDistanceMatrix, ~]   = loadObjectGivenFileName(files.nodesDistanceMatrixFile);

%% Files related to anomalies used for resampling
% Final anomalies values
[data.anomalies, ~]             = loadObjectGivenFileName(files.anomaliesFile);
% Anomalies on video
[data.anomaliesVideo, ~]        = loadObjectGivenFileName(files.anomaliesVideoFile);
% Anomalies on odometry
[data.anomaliesOdometry, ~]     = loadObjectGivenFileName(files.anomaliesOdometryFile);
% Anomalies on reconstruction
[data.anomaliesReconstruction, ~] = loadObjectGivenFileName(files.anomaliesReconstructionFile);
% KLDA anomaly
[data.KLDA, ~] = loadObjectGivenFileName(files.KLDAFile);
[data.KLDAs, ~] = loadObjectGivenFileName(files.KLDAsFile);

%% Files related to modification of the odometry vocabulary due to cutting
%  of clusters
[data.newClustersSequence, ~]          = loadObjectGivenFileName(files.newClustersSequenceFile);
[data.vectorOfKeptOdometryClusters, ~] = loadObjectGivenFileName(files.vectorOfKeptOdometryClustersFile);

%% Files related to restarting
[data.whenRestarted, ~]                = ...
    loadObjectGivenFileName(files.whenRestartedFile);
data.whenRestarted = data.whenRestarted + 1; % python starts from 0
[data.indicesRestartedParticles, ~]                = ...
    loadObjectGivenFileName(files.indicesRestartedParticlesFile);
data.indicesRestartedParticles = data.indicesRestartedParticles + 1;

end
