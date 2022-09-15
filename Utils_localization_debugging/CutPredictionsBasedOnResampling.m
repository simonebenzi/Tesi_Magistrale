
% Function to cut the hypothesis with resampling
% INPUTS:
% - predParams: original predictions of the parameters (odometry);
% - newIndicesForSwapping: indices of the resampling.
% OUTPUTS:
% - predParamsCorrected: predictions with cuttings through the resampling.
function [predParamsCorrected] = CutPredictionsBasedOnResampling(predParams, newIndicesForSwapping)

%numberOfParticles = size(predParams, 2);

len_predParams = size(predParams, 1);
len_newIndicesForSwapping = size(newIndicesForSwapping,1);
len_to_consider = min(len_predParams, len_newIndicesForSwapping);

indexOfLastResampling = 1;

% Initialize new vector of predicted params
predParamsCorrected = zeros(size(predParams,1),size(predParams,2), size(predParams,3));
% Looping over the number of time instans
for i = 1:len_to_consider

    % Check if resampling was performed in this epoch
    currentIndicesForSwapping = newIndicesForSwapping(i,:);
    wasThereResampling        = sum(currentIndicesForSwapping) ~= 0;

    % If resampling was done, eliminate those routes that were not 
    % resampled
    if wasThereResampling

        predParamsCorrected(indexOfLastResampling:i,:,:) = ...
            predParams(indexOfLastResampling:i,currentIndicesForSwapping+1,:);

        indexOfLastResampling = i;
    end
end

end