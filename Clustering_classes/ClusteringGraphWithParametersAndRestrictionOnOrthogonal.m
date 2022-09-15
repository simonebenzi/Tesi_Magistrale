
% CLASS for training the GNG, creating a Graph, or loading the Graph
% from a matlab file.
% In this case, the Graph includes the parameters of motion.

classdef ClusteringGraphWithParametersAndRestrictionOnOrthogonal < ClusteringGraphWithParameters
    
    %% PROPERTIES
    properties

        
    end
    
    %% METHODS
    methods
        
        %% Empty constructor
        function obj = ClusteringGraphWithParametersAndRestrictionOnOrthogonal()
            % Call ClusteringGraph constructor
            obj@ClusteringGraphWithParameters();
        end
        
        %% Find the distance of a point from the mean of the clusters
        % Inputs:
        % - x = datapoint
        % Outputs:
        % - d = distances from each cluster
        function [d] = FindDistancesOfPointFromClustersMean(obj, x)
            % Competion and Ranking
            innovation     = x - obj.nodesMean;
            
            % Now, decompose the error
            % For position
            positionInnovation   = innovation(:,2:3);
            projectionsPosition  = zeros(size(obj.nodesMeanStand,1), 2);
            orthogonalsDistance  = zeros(size(obj.nodesMeanStand,1), 2);
            for i = 1:size(obj.nodesMeanStand,1)
                pointVel = x(6:7);
                projectionsPosition(i,:) = GeometryHandler.ProjectPointOnLine2D( ...
                    pointVel, positionInnovation(i,:));
                distancePerp   = norm(projectionsPosition(i,:) - positionInnovation(i,:));
                distanceParall = norm(projectionsPosition(i,:));
                orthogonalsDistance(i,1) = distancePerp;
                orthogonalsDistance(i,2) = distanceParall;
            end
            % For speed
            velocityInnovation = innovation(:,6:7);
            % Combine
            innovation = [orthogonalsDistance, velocityInnovation];
            
            weights_repmat = repmat(obj.weightFeatures, size(obj.nodesMean, 1), 1);
            inn_by_weights = innovation.*weights_repmat;
            % Pairwise distance between two sets of observations, 
            % weighted by weight
            d              = mean(sqrt((inn_by_weights).^2), 2);
        end % end of FindDistancesOfPointFromStandardizedClustersMean function
        
        %% Function to find the closest cluster to a datapoint
        % Inputs:
        % - x: datapoint
        % Outputs:
        % - nodeIndex: index of closest cluster
        function [nodeIndex] = FindClosestCluster(obj, x)
            [d] = obj.FindDistancesOfPointFromClustersMean(x);
            [~, nodeIndex] = min(d); 
        end % end of FindClosestCluster function
        
        %% Find the distance of a point from the standardized mean of the clusters
        % Inputs:
        % - x = standardized datapoint
        % Outputs:
        % - d = distances from each cluster
        function [d] = FindDistancesOfPointFromStandardizedClustersMean(obj, xStand)
            % Competion and Ranking
            innovation     = xStand - obj.nodesMeanStand;
            % Now, decompose the error
            % For position
            positionInnovation   = innovation(:,2:3);
            projectionsPosition  = zeros(size(obj.nodesMeanStand,1), 2);
            orthogonalsDistance  = zeros(size(obj.nodesMeanStand,1), 2);
            for i = 1:size(obj.nodesMeanStand,1)
                pointVel = xStand(6:7);
                projectionsPosition(i,:) = GeometryHandler.ProjectPointOnLine2D( ...
                    pointVel, positionInnovation(i,:));
                distancePerp   = norm(projectionsPosition(i,:) - positionInnovation(i,:));
                distanceParall = norm(projectionsPosition(i,:));
                orthogonalsDistance(i,1) = distancePerp;
                orthogonalsDistance(i,2) = distanceParall;
            end
            % For speed
            velocityInnovation = innovation(:,6:7);
            % Combine
            innovation = [orthogonalsDistance, velocityInnovation];
            
            weights_repmat = repmat(obj.weightFeatures, size(obj.nodesMeanStand, 1), 1);
            inn_by_weights = innovation.*weights_repmat;
            % Pairwise distance between two sets of observations, 
            % weighted by weight
            d              = mean(sqrt((inn_by_weights).^2), 2);
        end % end of FindDistancesOfPointFromStandardizedClustersMean function
        
        %% Function to find the closest cluster to a standardized datapoint
        % Inputs:
        % - x: datapoint
        % Outputs:
        % - nodeIndex: index of closest cluster
        function [nodeIndex] = FindClosestClusterGivenStandardizedValues(obj, xStand)
            [d] = obj.FindDistancesOfPointFromStandardizedClustersMean(xStand);
            [~, nodeIndex] = min(d); 
        end % end of FindClosestClusterGivenStandardizedValues function
        
        %% Perform ranking
        % Inputs:
        % x: datapoint
        % Outputs:
        % s1: closest cluster
        % s2: second closest cluster
        % d: distances from clusters
        function [s1, s2, d] = PerformRanking(obj, x)
            % Competion and Ranking
            [d] = obj.FindDistancesOfPointFromStandardizedClustersMean(x);
            % Organize distances between nodes and the first data 
            % point in an ascending order                                       
            [~, SortOrder] = sort(d);                                           

            % Closest node index to the first data point
            s1 = SortOrder(1);    
            % Second closest node index to the first data point
            s2 = SortOrder(2);
        end % end of PerformRanking function

        %% Find features from data and already found nodesMeanStand
        function obj = ExtractClusteringFeaturesGivenDataAndNodesMeanStand(obj, ...
                                            parametersHolder, dataHolder, clusteringParameters, ...
                                            weightFeatures, nodesMeanStand)

            % Pass parameters holder
            obj.parametersHolder = parametersHolder;
            % Number of clusters
            obj.N                = numberOfClusters;

            % Input data from parameters holder
            inputData = parametersHolder.dataAndParametersInSingleArray;
            
            % Call ClusteringGraph function
            obj = ExtractClusteringFeaturesGivenDataAndNodesMeanStand@ClusteringGraph(obj, ...
                                            inputData, dataHolder, ...
                                            clusteringParameters, ...
                                            weightFeatures, nodesMeanStand);
        end

        %% Find features from data and already found cluster assignments
        function obj = ExtractClusteringFeaturesGivenDataAndClusterAssignments(obj, ...
                                            parametersHolder, dataHolder, clusteringParameters, ...
                                            weightFeatures, clusterAssignments)

            % Pass parameters holder
            obj.parametersHolder = parametersHolder;

            % Input data from parameters holder
            inputData = parametersHolder.dataAndParametersInSingleArray;
            % Call ClusteringGraph function
            obj = ExtractClusteringFeaturesGivenDataAndClusterAssignments@ClusteringGraph(...
                                            obj, inputData, dataHolder, ...
                                            clusteringParameters, ...
                                            weightFeatures, clusterAssignments);
        end
        
        %% Function to perform clustering
        % Overwritten from base to add parametersHolder
        function obj = PerformGNGClustering(obj, inputData, dataHolder, ...
                                            parametersHolder, ...
                                            clusteringParameters, ...
                                            weightFeatures)
            
            % Pass parameters holder
            obj.parametersHolder = parametersHolder;

            % Call ClusteringGraph function
            obj = PerformGNGClustering@ClusteringGraph(obj, inputData, dataHolder, ...
                                            clusteringParameters, ...
                                            weightFeatures);
        end % end of PerformGNGClustering function
    end
       
end