function [clusteringSubGraph] = ExtractSubGraphWithDirectCalculation(clusteringGraph, ...
    data, indicesOfSelectedFeatures)

dataSelection = data(:, indicesOfSelectedFeatures);
% General information
clusteringSubGraph.N                = clusteringGraph.N;
clusteringSubGraph.dataColorNode    = clusteringGraph.clusterAssignments;
clusteringSubGraph.startingPoints   = clusteringGraph.trajectoriesStartingPoints;
% Find datanodes of clusters
[clusteringSubGraph] = CalculateDatanodesGivenVocabulary(clusteringSubGraph, dataSelection);
% Initialize covariance
clusteringSubGraph.nodesCov         = cell(1, clusteringGraph.N);
clusteringSubGraph.nodesMean        = zeros(clusteringGraph.N, size(dataSelection,2));
% Extract features from datanodes
for i = 1:clusteringGraph.N
    clusteringSubGraph.nodesMean(i,:) = mean(clusteringSubGraph.datanodes{1,i},1);
    clusteringSubGraph.nodesCov{1, i} = cov(clusteringSubGraph.datanodes{1,i});
end

end