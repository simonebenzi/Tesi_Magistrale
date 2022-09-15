function [data] = LoadAllFilesDebugCode111(files, baseFolder, first_time_instant)

addpath('../../General_utils_code/')

%% General files
data.net                    = loadObjectGivenFileName(files.dictionaryFile);
data_sync                   = loadObjectGivenFileName(files.realOdometryFile);
if size(data_sync,1) < size(data_sync, 2)
    data_sync = data_sync';
end
data.data_sync              = data_sync(first_time_instant:end,:);

%% Run specific files
data.predictedParams_test           = squeeze(squeeze(loadObjectGivenFileName([baseFolder, '\', files.predictedParamsTestFile])));
data.clusterAssignmentsTest         = squeeze(loadObjectGivenFileName([baseFolder, '\', files.clusterAssignmentFile]));

odometryUpdatedParticles            = loadObjectGivenFileName([baseFolder, '\', files.updatedOdometriesOd]);
data.odometryUpdatedParticlesTestOd = permute(odometryUpdatedParticles, [1,3,2]);

data.newIndicesForSwapping          = loadObjectGivenFileName([baseFolder, '\', files.newIndicesForSwappingFile]);

end
