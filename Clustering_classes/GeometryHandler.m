
% Class with functions for handling geometry-related calculations.
% All methods are static

classdef GeometryHandler
    
    methods (Static)
        
        % Function to calculate the rotation center, knowing 
        % where an object was in two subsequent time instants, 
        % and the rotation angle.
        function [center_of_rot] = Find_rotation_center(curr_position, ...
                                                prev_position, prev_velocity, ...
                                                pred_position, pred_velocity, ...
                                                theta_curr_cluster)
                                            
            theta_curr_cluster = theta_curr_cluster + 1e-10;

            % Direction
            bending_direction       = pred_position - prev_position;
            % Point where to calculate the radius
            bending_point           = (pred_position + prev_position)/2;

            % Find the perpendicular to bending direction
            bending_direction_norm  = sqrt(bending_direction(1).^2 + ...
                                           bending_direction(2).^2) + 10^-16;
            bending_direction_norm  = bending_direction/bending_direction_norm;

            if sum(bending_direction_norm==0) == length(bending_direction_norm)
                perpendicular = bending_direction_norm;
            else
                perpendicular           = null(bending_direction_norm');
            end

            % Length of radius
            radius_len              = (norm(bending_direction)/2)/abs(theta_curr_cluster);

            center_of_rot           = bending_point + perpendicular*radius_len;

            % Find correct sign of perpendicular
            center_curr_pos         = norm(center_of_rot - curr_position);
            bendingPoint_curr_pos   = norm(center_of_rot - bending_point);
            if center_curr_pos < bendingPoint_curr_pos
                sign = -1;
                center_of_rot = bending_point + perpendicular*radius_len*sign;
            end
            
        end % end of Find_rotation_center function
        
        %% Function to project a point on a 3D plane
        function [X_projected] = ProjectPointOnPlane3D(velocity, orthogonal_to_cluster_mean)
            dimensions  = size(velocity, 2);
            A_point   = zeros(1,dimensions);
            for i = 1:dimensions-1
                dot_AP_AB(i)     = dot(orthogonal_to_cluster_mean(i,:), velocity);
                dot_AB_AB(i)     = dot(orthogonal_to_cluster_mean(i,:), orthogonal_to_cluster_mean(i,:));
                AB_line(i,:)     = orthogonal_to_cluster_mean(i,:);
                X_projected(i,:) = A_point + dot_AP_AB(i)./dot_AB_AB(i)*AB_line(i,:);
            end
        end % end of ProjectPointOnPlane3D function
        
        %% Function to project a point on a 2D plane
        function [X_projected] = ProjectPointOnLine2D(innovationDistance, point)
            dimensions = 2;
            A_point   = zeros(1,dimensions);
            for i = 1:dimensions-1
                dot_AP_AB(i)     = dot(innovationDistance(i,:), point);
                dot_AB_AB(i)     = dot(innovationDistance(i,:), innovationDistance(i,:));
                AB_line(i,:)     = innovationDistance(i,:);
                X_projected(i,:) = A_point + dot_AP_AB(i)./dot_AB_AB(i)*AB_line(i,:);
            end
        end
        
    end % end of methods
end