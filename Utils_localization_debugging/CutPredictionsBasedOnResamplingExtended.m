
% Function to cut the hypothesis with resampling.
% This function, conversely to the function
% 'CutPredictionsBasedOnResampling'
% does not only cut what results from the last resample, but goes back to
% the beginning.
% In extreme situations (in which at some point the particles all converge
% to a single hypothesis), this will result in a single particle being
% present for all the preceding history.
% INPUTS:
% - predParams: original predictions of the parameters (odometry);
% - newIndicesForSwapping: indices of the resampling.
% OUTPUTS:
% - predParamsCorrected: predictions with cuttings through the resampling.
function [predParamsCorrected] = CutPredictionsBasedOnResamplingExtended( ...
    predParams, newIndicesForSwapping, resamplesToDo)

%numberOfParticles = size(predParams, 2);

len_predParams = size(predParams, 1);
len_newIndicesForSwapping = size(newIndicesForSwapping,1);
len_to_consider = min(len_predParams, len_newIndicesForSwapping);

indexOfLastResampling        = 1;
indicesOfResamplingsUntilNow = [];

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

        if length(indicesOfResamplingsUntilNow) < resamplesToDo
            resampleBeginToConsider = 1;
        else
            resampleBeginToConsider = indicesOfResamplingsUntilNow(end-resamplesToDo+1);
        end

        predParamsCorrected(resampleBeginToConsider:indexOfLastResampling,:,:) = ...
            predParamsCorrected(resampleBeginToConsider:indexOfLastResampling,currentIndicesForSwapping+1,:);

        predParamsCorrected(indexOfLastResampling+1:i,:,:) = ...
            predParams(indexOfLastResampling+1:i,currentIndicesForSwapping+1,:);

        indexOfLastResampling        = i;
        indicesOfResamplingsUntilNow = [indicesOfResamplingsUntilNow; i];
    end
end

end