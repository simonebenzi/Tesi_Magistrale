
%% Check that we are moving in the direction of the next cluster
% We consider three cases to compare:
% A) Distance from point towards next cluster and next z point, as
%    of motion along z;
% B) Distance from point towards next cluster and next z point, as
%    of cluster prediction;
% C) Distance from point towards next cluster and next z point, as
%    of random prediction.
% D) Distance from center of next cluster and several steps forward
%    according to model motion rule.
% E) Distance from center of next cluster and no motion.

function [errors_nextPoint, errors_nextPointPred, ...
    errors_randomMotion, errors_nextPointPredSequence, errors_noMotion] = ...
    CalculateErrorsOfMotionTowardsNextCluster(...
    z_states, newClustersSequence, A_matrices, B_matrices, videoVocabulary)

flagEnd = false;

errors_nextPoint = [];
errors_nextPointPred = [];
errors_randomMotion = [];
errors_nextPointPredSequence = [];
errors_noMotion = [];

nextPointPreds = [];

for i = 1:size(z_states,1)-1
    
    %i
    
    %% Current point and next cluster
    % Current point on the z space
    currentPoint = z_states(i,:);
    % Current cluster
    currentCluster = newClustersSequence(i);
    % Finding the next cluster
    for j = 1:length(newClustersSequence)-i
        nextPossibleCluster = newClustersSequence(i+j);
        if currentCluster ~= nextPossibleCluster
            nextCluster = nextPossibleCluster;
            break;
        end
        if j == length(newClustersSequence)-i
            flagEnd = true;
        end
    end   
    % In case this is the end of the cluster sequence
    if flagEnd == true
        break;
    end
    % Next cluster mean state
    nextClusterCenter = videoVocabulary.nodesMeanstate(...
        nextCluster,:);
    % Versor towards next cluster
    versorTowardsNextCluster = nextClusterCenter - currentPoint;
    versorTowardsNextCluster = versorTowardsNextCluster/...
        norm(versorTowardsNextCluster);
    %% A)
    % Next point on the z space
    nextPoint = z_states(i+1,:);
    % Versor towards next motion
    versorTowardsNextPoint = nextPoint - currentPoint;
    versorTowardsNextPoint = versorTowardsNextPoint/...
        norm(versorTowardsNextPoint);
    % Error
    errorNextPoint = mean(abs(versorTowardsNextPoint - versorTowardsNextCluster));
    errors_nextPoint = [errors_nextPoint; errorNextPoint];
    %% B)
    % Predict from current state
    A_matrix_current = squeeze(A_matrices(i,:,:));
    B_matrix_current = squeeze(B_matrices(i,:));
    nextPointPred = A_matrix_current*currentPoint' + B_matrix_current';
    nextPointPred = nextPointPred';
    nextPointPreds = [nextPointPreds; nextPointPred];
    % Versor towards next motion
    versorTowardsNextPointPred = nextPointPred - currentPoint;
    versorTowardsNextPointPred = versorTowardsNextPointPred/...
        norm(versorTowardsNextPointPred);
    % Error
    errorNextPointPred = mean(abs(versorTowardsNextPointPred - versorTowardsNextCluster));
    errors_nextPointPred = [errors_nextPointPred; errorNextPointPred];
    %% C)
    % Select a random vector with values between -1 and 1
    randomZ = 2*rand(1,size(currentPoint,2))-1;
    % Versor towards random motion
    versorTowardsRandomMotion = randomZ/norm(randomZ);
    % Error
    errorRandomMotion = mean(abs(versorTowardsRandomMotion - versorTowardsNextCluster));
    errors_randomMotion = [errors_randomMotion; errorRandomMotion];
    %% D)
    % Max time in the cluster
    maxTimeInCluster = videoVocabulary.maxClustersTime(currentCluster);
    oneQuarterMaxTimeInCluster = ceil(maxTimeInCluster/4);
    % Project forward across several time steps
    pointProjected = currentPoint;
    for timeForward = 1:oneQuarterMaxTimeInCluster       
        pointProjected = A_matrix_current*pointProjected' + B_matrix_current';
        pointProjected = pointProjected';        
    end
    errorNextPoint = mean(abs(nextClusterCenter - pointProjected));
    errorNoMotion  = mean(abs(nextClusterCenter - currentPoint));
    errors_nextPointPredSequence = [errors_nextPointPredSequence; errorNextPoint];
    errors_noMotion = [errors_noMotion; errorNoMotion];
end

end