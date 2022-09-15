
% Function to calculate the transition matrix given the cluster assignments
% and given the total number of clusters.
% This can be used for example if transition matrix should have same
% dimension of training, but the data in testing is missing one or more
% of the last cluster assignments.
function [transitionMat] = CalculateTransitionMatrixGivenClusterAssignmentsAndMax(clusterAssignments, numberOfClusters)

% Total length of training data
currLength       = size(clusterAssignments, 1);
%  Transition matrix
transitionMat    = zeros(numberOfClusters,numberOfClusters);

%%%%%
for k = 1:currLength-1
    transitionMat(clusterAssignments(k,1),clusterAssignments(k+1,1)) =...
        transitionMat(clusterAssignments(k,1),clusterAssignments(k+1,1)) + 1;
end
transitionMat = transitionMat./repmat(sum(transitionMat,2) + (sum(transitionMat,2)==0),1,numberOfClusters);

end