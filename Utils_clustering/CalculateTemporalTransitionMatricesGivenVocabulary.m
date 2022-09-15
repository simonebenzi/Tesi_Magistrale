
% Input and Output: net structure obtained with GNG clustering
% The function creates a set of transition matrices, one for each time value
% from t to tMax, being tMax the time we have already spent in a node. 
function [net] = CalculateTemporalTransitionMatricesGivenVocabulary (net)

clusterAssignments = net.dataColorNode;

[transMatsTime]    = CalculateTemporalTransitionMatricesGivenClusterAssignments(clusterAssignments);

net.transMatsTime = transMatsTime;

end