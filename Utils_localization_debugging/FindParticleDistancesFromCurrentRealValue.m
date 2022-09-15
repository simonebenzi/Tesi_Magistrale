function [all_distances, min_distances, mean_distances] = ...
    FindParticleDistancesFromCurrentRealValue(real_odometry, predicted_odometry)

% Finding the absolute distance of all particles
% Finding the closest particle at each time instant

num_particles         = size(predicted_odometry,2);
num_time_instants     = min(size(real_odometry, 1), ...
                        size(predicted_odometry, 1));
all_distances         = zeros(num_time_instants, num_particles);
min_distances         = zeros(num_time_instants,1);
mean_distances        = zeros(num_time_instants,1);
indices_min_distances = zeros(num_time_instants,1);

for i = 1: num_time_instants    
    % Current real odometric value
    current_odometry_value = real_odometry(i,:);
    
    % All particles values at current time instant
    current_particles      = squeeze(predicted_odometry(i,:,:));
    
    % Calculating all mean absolute distances
    if num_particles > 1
        current_distances      = current_particles - ...
            repmat(current_odometry_value,num_particles,1);
        current_distances      = current_distances(:,1:2); % select position
        current_distances      = vecnorm(current_distances,2,2);
    else
        current_distances      = current_particles' - current_odometry_value;
        current_distances      = current_distances(1:2); % select position
        current_distances      = vecnorm(current_distances,2,2);
    end

    % Insert in vector with all distances
    all_distances(i,:)     = current_distances;
    
    % Min distance
    [min_value, min_index ]  = min(current_distances);
    min_distances(i)         = min_value;
    indices_min_distances(i) = min_index;
    mean_distances(i)        = mean(current_distances);
end


end