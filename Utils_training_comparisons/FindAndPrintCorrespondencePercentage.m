function [] = FindAndPrintCorrespondencePercentage(clusterAssignments_a, clusterAssignments_b)

lengthClusterAssignments = min(length(clusterAssignments_a), length(clusterAssignments_b));

clusterAssignments_a = clusterAssignments_a(1:lengthClusterAssignments);
clusterAssignments_b = clusterAssignments_b(1:lengthClusterAssignments);

numberOfSameCluster = sum(clusterAssignments_a == clusterAssignments_b);
numberOfSameCluster/lengthClusterAssignments

end