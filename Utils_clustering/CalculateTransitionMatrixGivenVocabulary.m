
% Input and Output: net structure obtained with GNG clustering
% This function calculates the overall transition matrix.
function [net] = CalculateTransitionMatrixGivenVocabulary(net)

clusterAssignments = net.dataColorNode;

[transitionMat]    = CalculateTransitionMatrixGivenClusterAssignments(clusterAssignments);

net.transitionMat  = transitionMat;

end