
% Function to calculate the transition matrix given the cluster
% assignments.
function [transitionMat] = CalculateTransitionMatrixGivenClusterAssignments(clusterAssignments)

% Number of clusters
numberOfClusters = max(clusterAssignments) - min(clusterAssignments) + 1;

[transitionMat]  = CalculateTransitionMatrixGivenClusterAssignmentsAndMax(clusterAssignments, numberOfClusters);

end