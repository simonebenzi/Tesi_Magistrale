
% Class to perform matching between TWO Graphs, called the source graph
% and the target graph.

classdef GraphMatcher
    
    %% PROPERTIES
    properties
        
        % Source and Target graphs
        graphSource
        graphTarget
        % Data of source and target graphs
        dataSource
        dataTarget
        % Number of clusters in source and target graphs
        clustersNumberSource
        clustersNumberTarget
        % Mean of training data in source and target graphs
        X_meanSource
        X_meanTarget
        % Standard deviation of training data in source and target graphs
        X_stdSource
        X_stdTarget
        % Mean values of clusters in source and target graphs
        nodeMeanSource
        nodeMeanTarget
        nodeMeanSourceStand
        nodeMeanTargetStand
        % Covariance values of clusters in source and target graphs
        nodeCovSource
        nodeCovTarget
        nodeCovSourceStand
        nodeCovTargetStand
        % Sequence of cluster assignments in source and target graphs
        clusterAssignmentsSource
        clusterAssignmentsTarget
        
    end
    
    methods
        
        %% Constructor
        % Takes two graphs and extracts the important information from it
        function obj = GraphMatcher(graphSource, graphTarget)
            obj.graphSource              = graphSource;
            obj.graphTarget              = graphTarget;
            
            obj.dataSource               = obj.graphSource.data;
            obj.dataTarget               = obj.graphTarget.data;
            
            obj.clustersNumberSource     = obj.graphSource.N;
            obj.clustersNumberTarget     = obj.graphTarget.N;
            obj.nodeMeanSource           = obj.graphSource.nodesMean;
            obj.nodeMeanTarget           = obj.graphTarget.nodesMean;
            obj.nodeCovSource            = obj.graphSource.nodesCov;
            obj.nodeCovTarget            = obj.graphTarget.nodesCov;
            obj.nodeMeanSourceStand      = obj.graphSource.nodesMeanStand;
            obj.nodeMeanTargetStand      = obj.graphTarget.nodesMeanStand;
            obj.nodeCovSourceStand       = obj.graphSource.nodesCovStand;
            obj.nodeCovTargetStand       = obj.graphTarget.nodesCovStand;
            obj.clusterAssignmentsSource = obj.graphSource.clusterAssignments;
            obj.clusterAssignmentsTarget = obj.graphTarget.clusterAssignments;
            
            obj.X_meanSource             = obj.graphSource.X_mean;
            obj.X_meanTarget             = obj.graphTarget.X_mean;
            obj.X_stdSource              = obj.graphSource.X_std;
            obj.X_stdTarget              = obj.graphTarget.X_std;
        end % end of constructor
        
        %% Function to find the cluster distances between the two graphs using a 
        % simple distance from the centers of the clusters
        % The varargin part includes the possibility of selecting which
        % features to use in the Matching. This allows to exclude features
        % that are potentially uninteresting for matching (e.g., position
        % could not always be relevant).
        % If it is not specified, all features are used.
        function minimumDistances = ...
                FindClusterDistancesAsSimpleEuclidean(obj, varargin)
            
            if nargin == 1
                % Find distance matrix using all the features of the mean
                selectedNodeMeanSource = obj.nodeMeanSource;
                selectedNodeMeanTarget = obj.nodeMeanTarget;
            else
                % Select features
                featuresToSelect = varargin{1};
                selectedNodeMeanSource = obj.nodeMeanSource(:, featuresToSelect);
                selectedNodeMeanTarget = obj.nodeMeanTarget(:, featuresToSelect);
            end
            % Find distance matrix selected features of the mean
            minimumDistances = GraphMatcher.FindMinClusterDistancesFromMeanMatrices(...
                selectedNodeMeanSource, selectedNodeMeanTarget);
        end
        % VERSION with standardized data and cluster centers
        function minimumDistances = ...
                FindClusterDistancesAsSimpleEuclideanStand(obj, varargin)
            
            if nargin == 1
                % Find distance matrix using all the features of the mean
                selectedNodeMeanSource = obj.nodeMeanSourceStand;
                selectedNodeMeanTarget = obj.nodeMeanTargetStand;
            else
                % Select features
                featuresToSelect = varargin{1};
                selectedNodeMeanSource = obj.nodeMeanSourceStand(:, featuresToSelect);
                selectedNodeMeanTarget = obj.nodeMeanTargetStand(:, featuresToSelect);
            end
            % Find distance matrix selected features of the mean
            minimumDistances = GraphMatcher.FindMinClusterDistancesFromMeanMatrices(...
                selectedNodeMeanSource, selectedNodeMeanTarget);
        end
        
        %% PLOTTING FUNCTIONS
        
        %% Function to plot the nodes of source graph more/less specific
        % to it with respect to target graph
        function PlotMinimumDistancesSourceGraph(obj, minimumDistances, positionsX, positionsY)
            % For coloring
            numberOfColors   = 256;
            myColorMap       = jet(numberOfColors);
            % Color per cluster
            colorIndexSource = (minimumDistances      - min(minimumDistances))/...
                               (max(minimumDistances) - min(minimumDistances));
            colorIndexSource = floor(colorIndexSource*255) + 1;
            
            % Loop over assigning distances per datapoint
            colorsIndexPerDataOfSource = obj.clusterAssignmentsSource;
            for k = 1 : length(minimumDistances)
                colorsIndexPerDataOfSource(obj.clusterAssignmentsSource==k) = ...
                    colorIndexSource(k);
            end
            % Plotting
            figure
            for k = 1 : length(minimumDistances)
                scatter(positionsX, positionsY, [], ...
                        myColorMap(colorsIndexPerDataOfSource, :));
            end
        end
        % This is called before 'PlotMinimumDistancesSourceGraph'
        % and supposes that positional data is present in the graph
        % Varargin:
        % 1 - minimumDistances
        % 2 (optional) - where position data is located. If this is not 
        %                assigned, it is supposed by default to be in 
        %                position 1 and 2
        function PlotMinimumDistancesSourceGraphWithPositionFromGraph(obj, varargin)
            % Distances for each cluster of source graph (w.r.t. clusters
            % of target graph.
            minimumDistances = varargin{1};
            % Retrieve where position is
            if nargin == 2
                posX = 1;
                posY = 2;
            elseif nargin == 4
                posX = varargin{2};
                posY = varargin{3};
            end
            positionsX = obj.dataSource(:,posX);
            positionsY = obj.dataSource(:,posY);
            % PLOTTING FUNCTION
            obj.PlotMinimumDistancesSourceGraph(minimumDistances, positionsX, positionsY)
        end
        % In this VERSION, we additionally give the position as input
        % instead of looking for it in the graph (as we could not have it
        % there)
        function PlotMinimumDistancesSourceGraphGivenPosition(obj, varargin)
            % Distances for each cluster of source graph (w.r.t. clusters
            % of target graph.
            minimumDistances = varargin{1};
            % Positions
            positionsX = varargin{2};
            positionsY = varargin{3};
            % PLOTTING FUNCTION
            obj.PlotMinimumDistancesSourceGraph(minimumDistances, positionsX, positionsY) 
        end % end of PlotMinimumDistancesSourceGraphGivenPosition function
    end % end of methods
    
    methods(Static)
        
        %% Function to find cluster distances given nodes mean matrices
        function minimumDistances = ...
                FindMinClusterDistancesFromMeanMatrices(...
                selectedNodeMeanSource, selectedNodeMeanTarget)
            
            % Find distance matrix selected features of the mean
             distancesBetweenNodesOfSourceAndTarget = ...
                    GraphMatcher.FindDistanceMatrixBetweenTwoMatrices(...
                    selectedNodeMeanSource, selectedNodeMeanTarget);
            % Pick the minimum distances for each cluster of source
            minimumDistances = min(distancesBetweenNodesOfSourceAndTarget,[],2);
        end % end of FindMinClusterDistancesFromMeanMatrices function
        
        %% Function to find distance matrix, given two matrices
        function clustersDistanceMatrix = ...
                FindDistanceMatrixBetweenTwoMatrices(matrixSource, matrixTarget)
            
            % Number of nodes of source and target graph
            clustersNumberSource      = size(matrixSource, 1);
            clustersNumberTarget      = size(matrixTarget, 1);
            % Initialize distance matrix
            clustersDistanceMatrix    = ...
                zeros(clustersNumberSource , clustersNumberTarget);
            % Actually find the distance matrix
            for i = 1:clustersNumberSource
                for j = 1:clustersNumberTarget
                    mean_s   = matrixSource(i, :);
                    mean_t   = matrixTarget(j, :);
                    distance = mean((mean_s - mean_t).^2);
                    % insert value in matrix
                    clustersDistanceMatrix(i, j) = distance; 
                end
            end
        end % end of FindDistanceMatrixBetweenTwoMatrices function
        
    end % end of static methods
end