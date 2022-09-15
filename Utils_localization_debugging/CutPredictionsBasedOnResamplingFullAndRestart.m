
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
function [predParamsCorrected] = CutPredictionsBasedOnResamplingFullAndRestart(...
    predParams, newIndicesForSwapping, whenRestarted, indicesRestartedParticles)

len_predParams = size(predParams, 1);
len_newIndicesForSwapping = size(newIndicesForSwapping,1);
len_to_consider = min(len_predParams, len_newIndicesForSwapping);

indexOfLastResampling = 1;

% Initialize new vector of predicted params
predParamsCorrected = zeros(size(predParams,1),size(predParams,2), size(predParams,3));
% Looping over the number of time instans
for i = 1:len_to_consider
    
    if i == 2128
        
        g = 3
        
    end

    % Check if resampling was performed in this epoch
    currentIndicesForSwapping = newIndicesForSwapping(i,:);
    wasThereResampling        = sum(currentIndicesForSwapping) ~= 0;
    wasThereRestarting        = sum(whenRestarted == i)  ~= 0;

    % If resampling was done, eliminate those routes that were not 
    % resampled
    if wasThereResampling

        predParamsCorrected(1:indexOfLastResampling,:,:) = ...
            predParamsCorrected(1:indexOfLastResampling,currentIndicesForSwapping+1,:);

        predParamsCorrected(indexOfLastResampling+1:i,:,:) = ...
            predParams(indexOfLastResampling+1:i,currentIndicesForSwapping+1,:);

        indexOfLastResampling = i;
        
        %% restarting case
        if wasThereRestarting
            
            reassignedParticles = [];
            
            particlesRestarted = indicesRestartedParticles(i,:);
            howManyRestartedParticles = size(indicesRestartedParticles,2);
            for j = 1:howManyRestartedParticles
                
                reassignedParticle = particlesRestarted(1);
                % Pick another particle before the restart
                while(sum(reassignedParticle==particlesRestarted)~= 0)
                    reassignedParticle = randi(size(predParamsCorrected,2));
                end
                reassignedParticles = [reassignedParticles;reassignedParticle];
            end      
            
            predParamsCorrected(1:i,particlesRestarted,:) = ...
                predParamsCorrected(1:i,reassignedParticles,:);

            %predParamsCorrected(indexOfLastResampling:i,particlesRestarted,:) = ...
            %predParams(indexOfLastResampling:i,reassignedParticles,:);
            
        end
    end
end

end