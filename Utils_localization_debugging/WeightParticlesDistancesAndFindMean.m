
function [summedDistancesOverTimeInstant, mean_distances] = WeightParticlesDistancesAndFindMean(all_distances, ...
    particles_weights)

distances_reweighted = all_distances.*particles_weights;
numberOfTimeInstants = size(distances_reweighted, 1);

summedDistancesOverTimeInstant = zeros(numberOfTimeInstants,1);

for i = 1:numberOfTimeInstants
    currentSumDistance = sum(distances_reweighted(i,:));
    summedDistancesOverTimeInstant(i) = currentSumDistance;
end

mean_distances = mean(summedDistancesOverTimeInstant);

end