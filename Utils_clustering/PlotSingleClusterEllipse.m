
function [] = PlotSingleClusterEllipse(clusterMean, clusterCovariance)

% First check the covariance matrix
clusterCovariance = CheckClusterCovarianceMatrix(clusterCovariance);
PlotErrorEllipse(clusterCovariance, clusterMean(1:2))

end