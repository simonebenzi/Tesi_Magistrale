
function [clusteringSubGraph] = ExtractSubGraphInSelectionOfFeatures(clusteringGraph, data, indicesOfSelectedFeatures)

clusteringSubGraph.N                = clusteringGraph.N;
clusteringSubGraph.dataColorNode    = clusteringGraph.clusterAssignments;
clusteringSubGraph.nodesMean        = clusteringGraph.nodesMean(:,indicesOfSelectedFeatures);
clusteringSubGraph.nodesCov       = cell(1, clusteringGraph.N);
clusteringSubGraph.nodesMeanStand = clusteringGraph.nodesMeanStand(:,indicesOfSelectedFeatures);
clusteringSubGraph.nodesCovStand    = cell(1, clusteringGraph.N);
clusteringSubGraph.X_mean           = clusteringGraph.X_mean(:,indicesOfSelectedFeatures);
clusteringSubGraph.X_std          = clusteringGraph.X_std(:,indicesOfSelectedFeatures);
clusteringSubGraph.startingPoints   = clusteringGraph.trajectoriesStartingPoints;
for i = 1:clusteringGraph.N
    clusteringSubGraph.nodesCov{1, i} = ...
        clusteringGraph.nodesCov{1,i}(indicesOfSelectedFeatures, indicesOfSelectedFeatures);
    clusteringSubGraph.nodesCovStand{1, i} = ...
        clusteringGraph.nodesCovStand{1,i}(indicesOfSelectedFeatures, indicesOfSelectedFeatures);
end
% Find datanodes of clusters
[clusteringSubGraph] = CalculateDatanodesGivenVocabulary(clusteringSubGraph, data(:, indicesOfSelectedFeatures));

end