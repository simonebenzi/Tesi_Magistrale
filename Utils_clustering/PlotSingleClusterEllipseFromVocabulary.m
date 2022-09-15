
function [] = PlotSingleClusterEllipseFromVocabulary(net, clusterIndex)

clusterDatanodes  = net.datanodes{1, clusterIndex};
scatter(clusterDatanodes(:,1), clusterDatanodes(:,2), 'r', 'filled', ...
        'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2)

clusterMean       = net.nodesMean(clusterIndex, :);
clusterCovariance = net.nodesCov{1, clusterIndex}(1:2, 1:2);
PlotSingleClusterEllipse(clusterMean, clusterCovariance)

end