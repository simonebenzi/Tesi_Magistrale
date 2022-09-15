
% Function to modify the odometry clustering in MATLAB according to how it
% had been changed online in python KVAE training, i.e., some clusters
% might have been eliminated.
% INPUTS:
% - net: clustering;
% - newClustersSequenceFile: sequence of new clusters;
% - vectorOfKeptOdometryClustersFile: a vector defining with 1 the clusters 
%   that were kept of the original clustering and with 0 the clusters that 
%   were eliminated.
% OUTPUTS:
% - net: the modified clustering that now might have some clusters less,
%   based on the newClustersSequence and vectorOfKeptOdometryClusters
%   files.
function [net] = ModifyOdometryClusteringBasedOnCutClusters(net, ...
    newClustersSequence, vectorOfKeptOdometryClusters)

% number of clusters
net.N = max(newClustersSequence) - min(newClustersSequence) + 1;
% If there are clusters to delete
if sum(vectorOfKeptOdometryClusters == 0) > 0
    % cluster assignments
    net.dataColorNode = newClustersSequence';
    if min(net.dataColorNode) == 0
        net.dataColorNode = net.dataColorNode + 1;
    end
    % Nodes mean
    net.nodesMean(vectorOfKeptOdometryClusters(1,:) == 0, :) = [];
    net.nodesMeanStand(vectorOfKeptOdometryClusters(1,:) == 0, :) = [];
    % Nodes cov
    net.nodesCov(:,vectorOfKeptOdometryClusters(1,:) == 0) = [];
    net.nodesCovStand(:,vectorOfKeptOdometryClusters(1,:) == 0) = [];
end

end