
function [alphas_from_assignments] = FindAlphasFromClusterAssignments( ...
    numberOfClusters, clusterAssignments)

alphas_from_assignments = zeros(size(clusterAssignments,1), numberOfClusters);

for i = 1:size(clusterAssignments,1)
    
    current_alphas_from_assignments = FindCurrentAlphaFromAssignment(...
        clusterAssignments(i,:),numberOfClusters);

    alphas_from_assignments(i, :) = current_alphas_from_assignments;
end

end

% Function to find the values of alpha at a certain time instant
% from the values of the cluster assignments at that time instant.
% INPUT: currentAssignments: cluster assignments at a time instant
% OUTPUT: current_alphas_from_assignments: probabilities of the clusters
%         at the time instant
function [current_alphas_from_assignments] = FindCurrentAlphaFromAssignment(...
    currentAssignments, numberOfClusters)

    current_alphas_from_assignments = zeros(numberOfClusters,1);
    for j = 1:size(currentAssignments,2)
        current_alphas_from_assignments(currentAssignments(j)+1) = ...
            current_alphas_from_assignments(currentAssignments(j)+1) + 1;
        current_alphas_from_assignments = ...
            current_alphas_from_assignments/sum(current_alphas_from_assignments);
    end
    
    current_alphas_from_assignments = current_alphas_from_assignments + 0.000000001;
    
end