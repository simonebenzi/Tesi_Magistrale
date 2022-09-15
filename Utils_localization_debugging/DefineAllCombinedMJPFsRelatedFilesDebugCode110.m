function [files] = DefineAllCombinedMJPFsRelatedFilesDebugCode110()

files.alphaTestFile           = 'OD_alpha_debugCode110.mat';
files.predictedParamsTestFile = 'OD_predictedParams_debugCode110.mat';
files.musTestFile             = 'OD_mus_debugCode110.mat';

files.clusterAssignmentFile   = 'OD_clusterAssignments_debugCode110.mat';

files.abn2File                = 'OD_videoMuAnomalies_debugCode110.mat';

files.clusterPassages         = 'OD_passagesBetweenClusters_debugCode110.mat';

files.probs2File              = 'OD_probs2_debugCode110.mat';
files.timeInClustersFile      = 'OD_timeInClusters_debugCode110.mat';

% Files related to odometric MJPF
files.predictedParamsTestOdFile = 'OD_predictedParams_od_debugCode110.mat';
files.timeInClustersOdFile      = 'OD_timeInClusters_od_debugCode110.mat';

files.updatedOdometriesOd       = 'OD_odometryUpdatedParticles_od_debugCode110.mat';
files.predictedOdometriesOd     = 'OD_odometryPredictedParticles_od_debugCode110.mat';

files.clusterAssignmentOdFile   = 'OD_clusterAssignments_od_debugCode110.mat';

files.probsReweightOdometries   = 'OD_probsReweigthOdometry_debugCode110.mat';
files.crossAssignmentsFile      = 'OD_indicesCrossAssignments_debugCode110.mat';

files.meanValuesAssignedFile    = 'OD_meanValuesAssigned_debugCode110.mat';

files.updatedValuesOdometryBeforeResampling         = ...
    'OD_updatedValuesOdometryBeforeResampling_debugCode110.mat';

files.mjpf_clustersToPickInResamplingFile           = ...
    'OD_mjpf_clustersToPick_numpy_debugCode110.mat';
files.mjpf_clusterProbabilitiesInResamplingFile     = ...
    'OD_mjpf_clusterProbabilities_numpy_debugCode110.mat';
files.mjpf_clusterProbabilities_newInResamplingFile = ...
    'OD_mjpf_clusterProbabilities_new_numpy_debugCode110.mat';
files.mjpf_indicesParticlesPerClusterInResamplingFile    = ...
    'OD_indicesParticlesPerCluster_numpy_debugCode110.mat';

files.very_first_mjpf_video_assignmentsFile    = ...
    'OD_very_first_mjpf_video_assignments_debugCode110.mat';
files.very_first_mjpf_odometry_assignmentsFile = ...
    'OD_very_first_mjpf_odometry_assignments_debugCode110.mat';

files.newIndicesForSwappingFile         = ...
    'OD_newIndicesForSwapping_numpy_debugCode110.mat';
files.newIndicesForSwappingReshapedFile = ...
    'OD_newIndicesForSwappingReshaped_numpy_debugCode110.mat';

files.clusterAssignmentsTrainFile = ...
    'OD_clusterAssignmentsTrain_debugCode110.mat';

files.whenResampledFile         = ...
    'OD_whenRestarted_numpy_debugCode110.mat';

files.particlesWeightsFile         = ...
    'OD_particlesWeights_debugCode110.mat';

%% Files related to updated and predicted and real a states
files.aStatesPredictionsFile = 'OD_aStatesPredictions_debugCode110.mat';
files.aStatesUpdatesFile     = 'OD_aStatesUpdates_debugCode110.mat';
files.aStatesRealFile        = 'OD_aStatesReal_debugCode110.mat';

%% Files related to the clustering
files.transitionMatFile         = 'OD_transitionMat_numpy_debugCode110.mat';
files.windowedtransMatsTimeFile = 'OD_windowedtransMatsTime_numpy_debugCode110.mat';
files.maxClustersTimeFile       = 'OD_maxClustersTime_numpy_debugCode110.mat';
files.nodesDistanceMatrixFile   = 'OD_nodesDistanceMatrix_numpy_debugCode110.mat';

%% Files related to anomalies used for resampling
files.anomaliesFile          = 'OD_anomalies_debugCode110.mat';
files.anomaliesVideoFile     = 'OD_anomalies_video_debugCode110.mat';
files.anomaliesOdometryFile  = 'OD_anomalies_odometry_debugCode110.mat';
files.anomaliesReconstructionFile  = 'OD_imageReconstructionAnomalies_debugCode110.mat';
files.KLDAFile  = 'OD_KLDA_debugCode110.mat';
files.KLDAsFile = 'OD_KLDAs_debugCode110.mat';

%% Files related to modification of the odometry vocabulary due to cutting
%  of clusters
files.newClustersSequenceFile          = 'newClustersSequence.mat';
files.vectorOfKeptOdometryClustersFile = 'vectorOfKeptOdometryClusters.mat';

%% Files related to restarting
files.whenRestartedFile         = ...
    'OD_whenRestarted_numpy_debugCode110.mat';
files.indicesRestartedParticlesFile         = ...
    'OD_indicesRestartedParticles_numpy_debugCode110.mat';

end