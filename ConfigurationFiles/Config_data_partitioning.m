
function [paramsTraining] = Config_data_partitioning()

% What ratio of the image data is used for training w.r.t. validation?
% Range: (0, 1].
paramsTraining.percentageOfTrajectoriesForTraining = 0.75;

end