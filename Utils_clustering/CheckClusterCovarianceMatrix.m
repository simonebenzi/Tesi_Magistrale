
function [clusterCovariance] = CheckClusterCovarianceMatrix(clusterCovariance)

    % First check the covariance matrix
    eigenvalues    = eig(clusterCovariance);
    isposdef       = all(eigenvalues > 0);
    sizeCovariance = size(clusterCovariance,1);
    
    if isposdef
    else
        clusterCovariance = real(diag(topdm(clusterCovariance))).*eye(sizeCovariance);
    end

end