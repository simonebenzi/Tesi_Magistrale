
% CLASS for training the GNG, creating a Graph, or loading the Graph
% from a matlab file.
% In this case, the Graph includes the parameters of motion.

classdef ClusteringGraphWithParameters < ClusteringGraph
    
    %% PROPERTIES
    properties
        
        % Parameters holder
        parametersHolder
        % Matrices A of each cluster
        rotationMatrices
        % Rotation centers of the clusters
        rotationCentersAllData
        rotationCentersAllDataPerCluster
        rotationCenters
        % Mean of clusters, only related to state
        nodesMeanStateOnly
        nodesMeanStateOnlyStand
        % Covariance of clusters, only related to state
        nodesCovStateOnly
        nodesCovStateOnlyStand
        % Mean norm of velocity of each cluster
        meanNormVel
        % Covariance of rotation centers
        covRotationCenters
        % Keeping data of clusters, only state
        dataNodesPerClusterStateOnly
        dataNodesPerClusterStateOnlyStand
        
        % From attempting testing with learned models on training
        predictedVelocitiesWithModel
        predictionErrorsWithModel
        predictionErrorsPerCluster
        
        % Cluster assignments by cell
        clusterAssignmentsCells
        
    end
    
    %% METHODS
    methods
        
        %% Empty constructor
        function obj = ClusteringGraphWithParameters()
            % Call ClusteringGraph constructor
            obj@ClusteringGraph();
        end
        
        %% Function to perform clustering
        % Overwritten from base to add parametersHolder
        function obj = PerformGNGClustering(obj, parametersHolder, dataHolder, ...
                                            clusteringParameters, ...
                                            weightFeatures)
            
            % Pass parameters holder
            obj.parametersHolder = parametersHolder;
            
            % Input data from parameters holder
            inputData = parametersHolder.dataAndParametersInSingleArray;
            % Call ClusteringGraph function
            obj = PerformGNGClustering@ClusteringGraph(obj, inputData, dataHolder, ...
                                            clusteringParameters, ...
                                            weightFeatures);
        end % end of PerformGNGClustering function
        
        %% Function to plot the clustering
        function Plot2DClustering(obj)
            % Take out positional data
            % We suppose for positional data to be placed at position
            % 1 and 2 of the data.
            posX = 2;
            posY = 3;
            positionalData = obj.data(:, [posX, posY]);
            
            % Create a color for each cluster
            colors     = parula(obj.N+1); % colors
            colorsData = colors(obj.clusterAssignments, :); % colors assigned to train data
            
            % Scatter
            figure
            scatter(positionalData(:,1), positionalData(:,2), [], colorsData);
            xlabel('x');
            ylabel('y');
            title('Clustering over positional data');
        end % end of Plot2DClustering function
        
        %% Function to plot the rotation centers of the clusters
        function PlotRotationCentersPerCluster(obj)
            figure
            % Looping over the clusters
            for cluster = 1: obj.N
                % Don't go above cluster 12 (for space reasons)
                if cluster <= 12
                    subplot(3, 4, cluster)
                    hold on
                    % Plot all points
                    scatter(obj.data(:,2), obj.data(:,3), 'k', 'filled'); 
                    % Plot points of current cluster
                    scatter(obj.dataNodesPerCluster{1, cluster}(:, 2), ...
                            obj.dataNodesPerCluster{1, cluster}(:, 3), 'r', 'filled');
                    % Plot centers of rotation of cluster
                    scatter(obj.rotationCenters(cluster,1), ...
                            obj.rotationCenters(cluster,2), 'g', 'filled');
                end
                % Add legend in first cluster only
                if cluster == 1
                    legend({'all data', 'cluster data', 'rot center'})
                end
            end
        end % end of PlotRotationCentersPerCluster function
        
        %% Function to extract the rotation matrices
        function obj = ExtractRotationMatricesPerCluster(obj)
            % Where theta is in the data
            thetaLocation = 1; % it is the first element of the data
            
            % Initialize rotation matrices
            obj.rotationMatrices = cell(1, obj.N) ;
            
            % Looping over the number of clusters
            for i = 1: obj.N
                
                % Rotation angle of current cluster
                thetaOfCurrentCluster        = obj.nodesMean(i,thetaLocation);
                
                % Build rotation matrix
                cos_A                        = cos(thetaOfCurrentCluster);
                sin_A                        = sin(thetaOfCurrentCluster);
                rotationMatrixCurrentCluster = [cos_A -sin_A; ...
                                                sin_A cos_A];
                                            
                % Assign to object
                obj.rotationMatrices{1, i}   = rotationMatrixCurrentCluster;

            end
        end % end of ExtractRotationMatricesPerCluster function
        
        %% Function to extract the datanodes per cluster, 
        % but only the part related to the state (pos+vel)
        function obj = ExtractDataNodesPerClusterStateOnly(obj)
            % Where state (x, vel) are located in the data
            stateLocation = [2, 3, 6, 7];
            % Initialize cells where to put the data of each cluster
            obj.dataNodesPerClusterStateOnly      = cell(1,obj.N);
            obj.dataNodesPerClusterStateOnlyStand = cell(1,obj.N);
            % Total number of training data
            lengthTrainingData      = size(obj.data, 1);
            
            % Finding the data in each cluster looping over the total data
            for time = 1:lengthTrainingData   
                
                % current data
                currDataPoint      = obj.data(time, stateLocation);
                currDataPointStand = obj.dataStand(time, stateLocation);
                
                % Cluster to which data 'dataPoint' belongs
                assignedCluster    = obj.clusterAssignments(time);

                % Assignment of the data point to its cluster cell
                % -- For original data
                obj.dataNodesPerClusterStateOnly{1,assignedCluster} = ...
                    [obj.dataNodesPerClusterStateOnly{1,assignedCluster}; currDataPoint];
                % -- For standardized data
                obj.dataNodesPerClusterStateOnlyStand{1,assignedCluster} = ...
                    [obj.dataNodesPerClusterStateOnlyStand{1,assignedCluster}; currDataPointStand];
            end
        end % end of dataNodesPerCluster function
        
        %% Extract the mean of the clusters, but only for state part (pos+vel)
        function obj = ExtractClustersMeanStateOnly(obj)
            % Initializing the mean
            obj.nodesMeanStateOnly = [];
            obj.nodesMeanStateOnlyStand = [];
            
            % Looping over the number of clusters
            for i = 1:obj.N
                
                % Calculation of mean values
                obj.nodesMeanStateOnly      = [obj.nodesMeanStateOnly; ...
                                      nanmean(obj.dataNodesPerClusterStateOnly{1,i},1)];
                obj.nodesMeanStateOnlyStand = [obj.nodesMeanStateOnlyStand; ...
                                      nanmean(obj.dataNodesPerClusterStateOnlyStand{1,i},1)];
            end
        end % end of ExtractClustersMean function
        
        %% Extract the covariance of the clusters, but only for state part (pos+vel)
        function obj = ExtractClustersCovStateOnly(obj)
            % Initialize covariances of clusters
            obj.nodesCovStateOnly      = cell(1, obj.N);
            obj.nodesCovStateOnlyStand = cell(1, obj.N);
            
            % Looping over the number of clusters
            for i = 1:obj.N
                obj.nodesCovStateOnly{1,i}       = nancov(obj.dataNodesPerClusterStateOnly{1,i});
                obj.nodesCovStateOnlyStand{1,i}  = nancov(obj.dataNodesPerClusterStateOnlyStand{1,i});
                
                % In case there was only one element in the cluster, 
                % create eye matrix, to avoid an empty covariance.
                if size(obj.nodesCovStateOnly{1,i}, 1) == 1
                    obj.nodesCovStateOnly{1,i}      = ...
                        eye(size(obj.dataNodesPerClusterStateOnly{1,i}, 2))*1e-25;
                    obj.nodesCovStateOnlyStand{1,i} = ...
                        eye(size(obj.dataNodesPerClusterStateOnlyStand{1,i}, 2))*1e-25;
                end
            end
        end % end of ExtractClustersCov function
        
        %% Extract the rotation centers of the clusters
        function obj = ExtractRotationCentersOfClusters(obj)
            % Initialize rotation centers
            obj.rotationCenters      = zeros(obj.N, 2);
            % Initialize covariance of rotation centers
            obj.covRotationCenters   = cell(obj.N, 1);
            
            % Looping over the clusters
            for cluster = 1: obj.N

                % Find the center
                obj.rotationCenters(cluster,:) = ...
                        mean(obj.rotationCentersAllDataPerCluster{cluster, 1} , 1);
                    
                % Find center covariance
                obj.covRotationCenters{cluster, 1} = ...
                        cov(obj.rotationCentersAllDataPerCluster{cluster, 1} , 1);
            end
        end % end of ExtractRotationCentersOfClusters function
        
        %% Extract normalized velocity of clusters
        function obj = ExtractVelNormCentersOfClusters(obj)
            obj.meanNormVel = zeros(obj.N, 2);
            
            % Looping over the clusters
            for cluster = 1: obj.N
            
                % Find the norm of mean vel (so it is 1D)
                v_x                   = obj.dataNodesPerCluster{1,cluster}(:, 6);
                v_y                   = obj.dataNodesPerCluster{1,cluster}(:, 7);
                velocityNormOfCluster = sqrt(v_x.^2 + v_y.^2);

                obj.meanNormVel(cluster)  = mean(velocityNormOfCluster);
            end
        end % end of ExtractVelNormCentersOfClusters function
        
        %% Extract rotation centers over all data
        function obj = ExtractRotationCentersOfAllData(obj)
            % Vector of all rotation centers, for all data points
            obj.rotationCentersAllData = [];
            % Cells of all rotation centers, for all data points, divided
            % by cluster.
            obj.rotationCentersAllDataPerCluster = cell(obj.N, 1);
            % Total length of training data
            dataLength        = size(obj.data,1); 

            % Looking for center for each cluster
            for i = 1: dataLength

                % Assigned cluster of current data point
                clusterOfCurrentDataPoint = obj.clusterAssignments(i);
                
                % Previous position and velocity
                if i > 1
                    positionOfPreviousDataPoint = obj.data(i-1, 2:3)';
                    velocityOfPreviousDataPoint = obj.data(i-1, 6:7)';
                else
                    positionOfPreviousDataPoint = obj.data(i, 2:3)';
                    velocityOfPreviousDataPoint = obj.data(i, 6:7)';
                end
                
                % Transition matrix of current cluster
                matrixAOfCurrentDataPoint = ...
                    obj.rotationMatrices{1, clusterOfCurrentDataPoint};

                % Theta of current cluster
                thetaOfCurrentDataPoint   = ...
                    obj.nodesMean(clusterOfCurrentDataPoint,1);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                % Perform prediction: previous -> current
                [positionOfCurrentDataPoint, velocityOfCurrentDataPoint] = ...
                    ClusteringGraphWithParameters.PredictWithRotationMatrix( ...
                     positionOfPreviousDataPoint, velocityOfPreviousDataPoint, ...
                     matrixAOfCurrentDataPoint);
                 
                 % Perform prediction: current -> next
                 [positionOfNextDataPoint, velocityOfNextDataPoint] = ...
                    ClusteringGraphWithParameters.PredictWithRotationMatrix( ...
                     positionOfCurrentDataPoint, velocityOfCurrentDataPoint, ...
                     matrixAOfCurrentDataPoint);
                 
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                % Find rotation center
                [rotationCenterOfCurrentDataPoint] = ...
                    GeometryHandler.Find_rotation_center(...
                     positionOfCurrentDataPoint,positionOfPreviousDataPoint, ...
                     velocityOfPreviousDataPoint, ...
                     positionOfNextDataPoint, velocityOfNextDataPoint, ...
                     thetaOfCurrentDataPoint);

                % Centers of rotation saving
                obj.rotationCentersAllData = [obj.rotationCentersAllData; ...
                                              rotationCenterOfCurrentDataPoint'];
                                      
                obj.rotationCentersAllDataPerCluster{clusterOfCurrentDataPoint, 1} = ...
                    [obj.rotationCentersAllDataPerCluster{clusterOfCurrentDataPoint, 1}; ...
                     rotationCenterOfCurrentDataPoint'];
            end
        end % end of ExtractRotationCentersOfAllData function
        
        %% Extract cluster assignment per cells
        function obj = ExtractClusterAssignmentsByCells(obj)
            % How many data cells in the data
            dataCellNumbers = size(obj.dataHolder.Data, 1);
            % Initialize cluster assignments
            obj.clusterAssignmentsCells = cell(dataCellNumbers, 1);
            
            count = 1;
            for i = 1: dataCellNumbers
                % How many elements are in the current cell
                sizeOfCurrentCell                 = size(obj.dataHolder.Data{i, 1}, 1);
                % Find the cluster assignments for this cell
                obj.clusterAssignmentsCells{i, 1} = obj.clusterAssignments( ...
                                                      count: count + sizeOfCurrentCell - 1);                              
                % Increment count of data considered
                count                             = count + sizeOfCurrentCell;
            end
        end % end of ExtractClusterAssignmentsByCells function
        
        %% Extract clustering features
        function obj = ExtractClusteringFeatures(obj)
            % Call base one
            obj = ExtractClusteringFeatures@ClusteringGraph(obj);
            
            % Extract rotation matrix
            obj = obj.ExtractRotationMatricesPerCluster();
            % Extract datapoints in each cluster, considering only state
            obj = obj.ExtractDataNodesPerClusterStateOnly();
            
            % Extract mean and covariance for state only 
            obj = obj.ExtractClustersMeanStateOnly();
            obj = obj.ExtractClustersCovStateOnly();
            
            % Extract rotation centers over all the data
            obj = obj.ExtractRotationCentersOfAllData();
            % Extract rotation centers per cluster
            obj = obj.ExtractRotationCentersOfClusters();
            % Extract velocity norm per cluster
            obj = obj.ExtractVelNormCentersOfClusters();
            
            % Extract cluster assignments by cells
            % Extract cells containing cluster assignments
            obj = obj.ExtractClusterAssignmentsByCells();
        end % end of ExtractClusteringFeatures function
       
    end % end of methods
    
    methods (Static)
        
        %% Perform prediction using rotation matrix
        function [predictedPosition, predictedVelocity] = PredictWithRotationMatrix( ...
                position, velocity, rotationMatrix)
            
            predictedVelocity = rotationMatrix*velocity;
            predictedPosition = position + predictedVelocity;
        end
    end % end of static methods 
end