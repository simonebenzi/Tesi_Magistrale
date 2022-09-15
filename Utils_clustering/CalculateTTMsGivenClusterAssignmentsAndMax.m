
% Input and Output: net structure obtained with GNG clustering
% The function creates a set of transition matrices, one for each time value
% from t to tMax, being tMax the time we have already spent in a node. 
function [transMatsTime] = CalculateTTMsGivenClusterAssignmentsAndMax(clusterAssignments, numberOfClusters)

% Total length of training data
currLength  = size(clusterAssignments, 1);

% Find max number of time instants before a zone change 
tMax = Find_Max_Time_Before_Transition(clusterAssignments);

% Find TIME TRANSITION MATRICES for follower
transMatsTime = Find_Time_Transition_Matrices(tMax, ...
    numberOfClusters, clusterAssignments, currLength);

end