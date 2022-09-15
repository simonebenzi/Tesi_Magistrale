
% Function to find to which cluster (given the means of the clusters) would
% be assigned each datapoint of a set. Simple MSE is used.
% INPUTS:
% - data: datapoints -> (number_of_time_instants, data_dimension)
% - nodesMean: mean values of the clusters -> (number_of_clusters,
%              data_dimension)
function [clusterAssignments] = FindClusterAssignmentsOfDatapoints (data, nodesMean)

clusterAssignments = [];
for i = 1: size(data, 1)
    currDataAndParameters = data(i, :);
    innovation = currDataAndParameters - nodesMean;
    d = mean(sqrt((innovation).^2), 2);
    [~, nodeIndex] = min(d); 
    clusterAssignments          = [clusterAssignments; nodeIndex];
end

end