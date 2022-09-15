
function [paramsTraining] = Config_filtering()

%% Filtering covariances
% Observation, prediction and initial variance for the Kalman Filter
paramsTraining.obsVar              = 0.05;
paramsTraining.predVar             = 0.5;
paramsTraining.initialVar          = 0.1;

%% Memory window
% A smoothing is performed using a memory of this length
paramsTraining.memoryLength        = 4;
% Weights for defining how important is the closer data in time, and the
% further away. Here it is defined so that all data in the window of time
% has equal importance.
paramsTraining.memoryWeights       = ones(1, paramsTraining.memoryLength);
paramsTraining.memoryWeights       = paramsTraining.memoryWeights/paramsTraining.memoryLength;

end