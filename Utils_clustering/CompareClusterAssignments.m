
function [clusterAssignments] = CompareClusterAssignments(numberOfClusters, ...
    clusterAssignmentsOriginal, statesAdditional, nodesMeanAdditional, ...
    odometry)

% Colors for plotting
colors = parula(numberOfClusters); 
colorsOdometryGraph = colors(clusterAssignmentsOriginal, :); 
% Find cluster assignments of video
[clusterAssignments] = FindClusterAssignmentsOfDatapoints (statesAdditional, ...
    nodesMeanAdditional);
% Plotting the odometry assignments
figure
subplot(1,2,1)
scatter(odometry(:, 1), odometry(:, 2), [], colorsOdometryGraph);
title('Cluster assignments of Odometry')
% Plotting the video assignments
subplot(1,2,2)
scatter(odometry(1:length(clusterAssignments), 1), ...
        odometry(1:length(clusterAssignments), 2), [], clusterAssignments);
title('Cluster assignments of Video')
% Ratio of same assignments
numberOfSameCluster = sum(clusterAssignments == ...
    clusterAssignmentsOriginal(1:length(clusterAssignments)));
disp('Ratio of corresponding assignments using odometry and video')
numberOfSameCluster/length(clusterAssignments)

end