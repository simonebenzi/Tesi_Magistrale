
% Function to perform video clustering using the odometry cluster
% assignments.
% INPUTS:
% - a_states: a_states from last epoch of KVAE training
% - z_states: z_states from last epoch of KVAE training
% - alphas: alphas from last epoch of KVAE training
% OUTPUTS:
% - net: the video clustering, which is also saved as a .mat file in
%   'folder' with the name of 'Odometry_based_vocabulary.mat'.
function [net] = PerformClusteringWithOdometryAssignments(a_states, z_states, alphas)

%% Addpath
addpath('../../General_utils_code')
%% Number of clusters
N          = size(alphas,2);
dataLength = size(a_states, 1);
%% Find cluster assignments using the alphas
for i = 1:dataLength
   [~, max_index] = max(alphas(i,:)); 
   dataColorNode(i,1) = max_index;
end
%% Vocabulary means and covs
% Divide data by cluster
datanodes_state_a = cell(1, N);
datanodes_state_z = cell(1, N);
for i = 1 : dataLength
   currCluster = dataColorNode(i);
   datanodes_state_a{1, currCluster} = ...
       [datanodes_state_a{1, currCluster}; a_states(i, :)];
   datanodes_state_z{1, currCluster} = ...
       [datanodes_state_z{1, currCluster}; z_states(i, :)];
end
% Filling the means and covariances
nodesMean_a_states = [];
nodesMean_z_states = [];
for i = 1:size(datanodes_state_a,2)
    %   Calculation of mean values
    nodesMean_a_states         = [nodesMean_a_states; ...
                                  mean(datanodes_state_a{1,i}, 1)];
    nodesMean_z_states         = [nodesMean_z_states; ...
                                  mean(datanodes_state_z{1,i}, 1)];    
    %   Calculation of covariance values
    nodesCov_a_states{1,i}     = cov(datanodes_state_a{1,i}, 1);
    nodesCov_z_states{1,i}     = cov(datanodes_state_z{1,i}, 1);    
    if isnan(nodesCov_a_states{1,i})
        nodesCov_a_states{1,i} = zeros(size(a_states,2), size(a_states,2));
        nodesCov_z_states{1,i} = zeros(size(z_states,2), size(z_states,2));
        nodesMean_a_states     = [nodesMean_a_states; ...
                                  zeros(1,size(a_states,2))];
        nodesMean_z_states     = [nodesMean_z_states; ...
                                  zeros(1,size(z_states,2))];
    end
end
%% Fill the vocabulary
net.dataColorNode = dataColorNode;
net.dataColorDirectAssignment = dataColorNode;
net.nodesMean = nodesMean_a_states;
net.nodesMeanstate = nodesMean_z_states;
net.nodesCov  = nodesCov_a_states;
net.nodesCovstate  = nodesCov_z_states;
net.N                  = N;
net.a_states           = a_states;
net.z_states           = z_states;
net.alphas             = alphas;
net.data               = z_states;
%% Calculate transition matrix
net = CalculateTransitionMatrix(net);
%% Calculate temporal transition matrix
net = CalculateTemporalTransitionMatrices (net);
%% Find the max time spent in each cluster
net = CalculateMaxClustersTime (net);
end