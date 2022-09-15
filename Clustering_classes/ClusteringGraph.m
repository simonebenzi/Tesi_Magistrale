
% CLASS for training the GNG, creating a Graph, or loading the Graph
% from a matlab file.

classdef ClusteringGraph
    
    %% PROPERTIES
    properties
        
        % DataHolder object
        dataHolder
        % Training data
        data
        dataStand % Standardized data
        % Mean of training data
        X_mean
        % Std of training data
        X_std
        % Cells, each containing the data assigned to that cluster
        dataNodesPerCluster
        dataNodesPerClusterStand % Standardized version
        % Sequence of cluster assignments for training data
        clusterAssignments
        
        % Number of clusters
        N
        % Mean of the nodes
        nodesMean
        % Mean of the nodes, standardized
        nodesMeanStand
        % Covariance of the nodes
        nodesCov
        % Covariance of the nodes, standardized
        nodesCovStand
        
        % Transition matrix
        transitionMat
        % Temporal Transition matrices
        temporalTransitionMats
        % Max time spent in a cluster, overall
        maxOverallClusterTime
        % Max time spent in each cluster
        maxClustersTime
        % Time in each cluster
        timeInEachCluster
        
        % Parameters used in clustering
        clusteringParameters
        % Weights to give to the different features
        weightFeatures

        % Beginning points of trajectories
        trajectoriesStartingPoints
        
        % History 
        meanErrorsInTraining
        numberOfNeuronsInTraining
        
        % GNG
        E 
        utility
        C
        t

    end
    
    %% METHODS
    methods
        
        %% Empty constructor
        function obj = ClusteringGraph()
            
        end % end of constructor
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Function for displaying the Graph
        function Plot2DClustering(obj, positionalData)
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
        %% Function for displaying the Graph on temporal dimension
        function PlotClusteringAlongTime(obj)
            % Create a color for each cluster
            colors     = parula(obj.N+1); % colors
            colorsData = colors(obj.clusterAssignments, :); % colors assigned to train data
            
            % Scatter
            figure
            scatter(1:1:size(colorsData, 1), ...
                    ones(1, size(colorsData, 1)), [], colorsData);
            xlabel('time');
            title('Clustering over positional data');
        end % end of Plot2DClustering function
        %% Function to plot the clustering using the found clustering
        % assignments and supposing that position was in the data
        function PlotClusteringHavingPositionInClusteringData(obj)
            % Take out positional data
            % We suppose for positional data to be placed at position
            % 1 and 2 of the data.
            posX = 1;
            posY = 2;
            positionalData = obj.data(:, [posX, posY]);
            % Plotting
            obj.Plot2DClustering(positionalData)
        end % end of PlotClusteringHavingPositionInClusteringData function
        %% Function to plot the clustering using the found clustering
        % assignments and supposing to be given the corresponding positions
        function PlotClusteringGivenCorrespondingPositions(obj, positionalData)
            obj.Plot2DClustering(positionalData)
        end % end of PlotClusteringGivenCorrespondingPositions function
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Functions for extracting the Graph information once the 
        %  clustering has been performed
        
        %% Function to extract the cluster mean
        function obj = ExtractClusterMean(obj)           
            obj.X_mean = nanmean(obj.data, 1);
        end % end of ExtractClusterMean function
        
        %% Function to extract the cluster standard deviation
        function obj = ExtractClusterStd(obj)            
            obj.X_std = nanstd(obj.data, 1);
        end % end of ExtractClusterStd function
        
        %% Function to extract the datanodes per cluster
        function obj = ExtractDataNodesPerCluster(obj)
            % Initialize cells where to put the data of each cluster
            obj.dataNodesPerCluster      = cell(1,obj.N);
            obj.dataNodesPerClusterStand = cell(1,obj.N);
            
            % Total number of training data
            lengthTrainingData      = size(obj.data, 1);
            
            % Finding the data in each cluster looping over the total data
            for time = 1:lengthTrainingData   
                
                % current data
                currDataPoint = obj.data(time, :);
                currDataPointStand = obj.dataStand(time, :);
                
                % Cluster to which data 'dataPoint' belongs
                assignedCluster = obj.clusterAssignments(time);

                % Assignment of the data point to its cluster cell
                % -- For original data
                obj.dataNodesPerCluster{1,assignedCluster} = ...
                    [obj.dataNodesPerCluster{1,assignedCluster}; currDataPoint];
                % -- For standardized data
                obj.dataNodesPerClusterStand{1,assignedCluster} = ...
                    [obj.dataNodesPerClusterStand{1,assignedCluster}; currDataPointStand];
            end
        end % end of dataNodesPerCluster function
        
        %% Function to extract the means of the clusters
        function obj = ExtractClustersMean(obj)
            % Initializing the mean
            obj.nodesMean = [];
            obj.nodesMeanStand = [];
            
            % Looping over the number of clusters
            for i = 1:obj.N
                
                % Calculation of mean values
                obj.nodesMean      = [obj.nodesMean; ...
                                      nanmean(obj.dataNodesPerCluster{1,i},1)];
                obj.nodesMeanStand = [obj.nodesMeanStand; ...
                                      nanmean(obj.dataNodesPerClusterStand{1,i},1)];
            end
        end % end of ExtractClustersMean function
        
        %% Function to extract the covariances of the clusters
        function obj = ExtractClustersCov(obj)
            % Initialize covariances of clusters
            obj.nodesCov      = cell(1, obj.N);
            obj.nodesCovStand = cell(1, obj.N);
            
            % Looping over the number of clusters
            for i = 1:obj.N
                obj.nodesCov{1,i}       = cov(obj.dataNodesPerCluster{1,i});
                obj.nodesCovStand{1,i}  = cov(obj.dataNodesPerClusterStand{1,i});
                
                % In case there was only one element in the cluster, 
                % create eye matrix, to avoid an empty covariance.
                if size(obj.nodesCov{1,i}, 1) == 1
                    obj.nodesCov{1,i}      = ...
                        eye(size(obj.dataNodesPerCluster{1,i}, 2))*1e-25;
                    obj.nodesCovStand{1,i} = ...
                        eye(size(obj.dataNodesPerClusterStand{1,i}, 2))*1e-25;
                end
            end
        end % end of ExtractClustersCov function
        
        %% Function to extract the transition matrix
        function obj = ExtractTransitionMatrix(obj)
            % Initialize transition matrix to zero
            obj.transitionMat = zeros(obj.N);
            
            % Total length of training data
            dataLength        = size(obj.data,1); 

            % Starting point of trajectories
            startingPoints    = obj.trajectoriesStartingPoints;
            
            % Increment by +1 every time of a transition between two clusters
            for k = 1:dataLength - 1
                if sum(startingPoints == k+1) == 0
                    obj.transitionMat(obj.clusterAssignments(k,1),obj.clusterAssignments(k+1,1)) =...
                        obj.transitionMat(obj.clusterAssignments(k,1),obj.clusterAssignments(k+1,1)) + 1;
                end
            end

            % Normalising the transition matrix so that the sum of the rows gives 1
            obj.transitionMat = obj.transitionMat./ ...
                repmat(sum(obj.transitionMat,2) + (sum(obj.transitionMat,2)==0),1,obj.N);
        end % end of ExtractTransitionMatrix function
        
        %% Find the maximum time spent in a cluster, considering all
        % clusters
        function[obj] = ExtractMaxOverallTimeBeforeTransition(obj)
            % Total length of training data
            dataLength        = size(obj.data,1); 
            % Initialise the max to 1
            obj.maxOverallClusterTime = 1;
            % Initialise the variable containing the previous zone of a comparison
            % with the first zone of the trajectory
            oldZone = obj.clusterAssignments(1,1);
            % Initialise the length of the run
            currentMax = 1;
            
            % Looping over all the training data points
            for t = 2 : dataLength
                % Zone at current time
                newZone = obj.clusterAssignments(t);

                % If the zone has not changed
                if newZone == oldZone
                    % Increment the max of current run
                    currentMax = currentMax + 1;
                % if the zone has changed
                else
                    oldZone = newZone;
                    % Check if a longer run has been foung
                    if currentMax > obj.maxOverallClusterTime
                        obj.maxOverallClusterTime = currentMax;
                    end
                    % Reinitialize current max
                    currentMax = 1;
                end  
            end
        end % end of FindMaxOverallTimeBeforeTransition function
        
        %% Function to extract the temporal transition matrices
        function obj = ExtractTemporalTransitionMatrices(obj)
            % Total length of training data
            dataLength        = size(obj.data,1); 
            % Initialize the temporal transition matrices
            obj.temporalTransitionMats = cell(1, obj.maxOverallClusterTime);

            % Starting point of trajectories
            startingPoints    = obj.trajectoriesStartingPoints;
            
            % Initialize a number of 'maxOverallClusterTime' transition
            % matrices, each of dimension NxN
            for i = 1: obj.maxOverallClusterTime
                obj.temporalTransitionMats{1, i} = zeros(obj.N);
            end

            % Take the first zone in the trajectory
            prevZone = obj.clusterAssignments(1, 1);
            time = 0;

            % loop over the number of time instants of the trajectory
            for t=2:dataLength
                % I add a time instant
                time = time + 1;
                if (time > obj.maxOverallClusterTime)
                    break
                end
                % New zone
                newZone = obj.clusterAssignments(t);
                
                if sum(startingPoints == t) == 0
                    % If I change the zone with respect to the previous time
                    % instant(/s)
                    if (prevZone ~= newZone)
                        % I increment the corresponding value in the correct transition
                        % matrix
                        obj.temporalTransitionMats{1,time}(prevZone, newZone) = ...
                            obj.temporalTransitionMats{1,time}(prevZone, newZone) + 1;
                        % And I update the zone value
                        prevZone = newZone;    
                        % I reinitialize the time
                        time = 0;
                    % Otherwise, if I remain in the same zone
                    else
                        obj.temporalTransitionMats{1,time}(prevZone, prevZone) = ...
                            obj.temporalTransitionMats{1,time}(prevZone, prevZone) + 1;
                    end
                else
                    % And I update the zone value
                    prevZone = newZone;
                    % I reinitialize the time
                    time = 0;
                end
            end

            % For each transition matrix
            for t = 1:obj.maxOverallClusterTime
                % looping over rows of current matrix
                for row= 1: obj.N
                    % sum of the elements of the row
                    sumElementsRow = sum(obj.temporalTransitionMats{1, t}(row, :));
                    % to prevent division by 0
                    if sumElementsRow ~=0
                        % looping over columns of current matrix
                        for column = 1: obj.N
                            % normalise matrix element
                            obj.temporalTransitionMats{1,t}(row, column) = ...
                                obj.temporalTransitionMats{1,t}(row, column)/sumElementsRow;
                        end
                    end
                end
            end
        end % end of ExtractTemporalTransitionMatrices function
        
        %% Function to extract the maximum time spent in EACH cluster
        function obj = ExtractMaxClustersTime(obj)
            % Total length of training data
            dataLength        = size(obj.data,1); 
            % Where to insert the max time values
            obj.maxClustersTime = zeros(1, obj.N);
            % Initialise the variable containing the previous zone of a comparison
            % with the first zone of the trajectory
            oldZone = obj.clusterAssignments(1,1);
            % Initialise the length of the run
            currentMax = 1;
            
            % Loop over the data
            for t = 1: dataLength
                % Zone at current time
                newZone = obj.clusterAssignments(t);

                % If the zone has not changed
                if newZone == oldZone
                    % Increment the max of current run
                    currentMax = currentMax + 1;
                % if the zone has changed
                else
                    % Check if a longer run has been foung
                    if currentMax > obj.maxClustersTime(oldZone)
                        obj.maxClustersTime(oldZone) = currentMax;
                    end
                    oldZone = newZone;
                    % Reinitialize current max
                    currentMax = 1;
                end  
            end
        end % end of ExtractMaxClustersTime function

        %% Function to extract the maximum time spent in EACH cluster
        function obj = ExtractTimeInEachCluster(obj)
            % Total length of training data
            dataLength        = size(obj.data,1); 
            % Initialise the variable containing the previous zone of a comparison
            % with the first zone of the trajectory
            oldZone = obj.clusterAssignments(1,1);

            % Times spent in each cluster
            obj.timeInEachCluster = cell(obj.N, 1);

            % Initialise the length of the run
            currentMax = 1;
            
            % Loop over the data
            for t = 1: dataLength
                % Zone at current time
                newZone = obj.clusterAssignments(t);

                % If the zone has not changed
                if newZone == oldZone
                    % Increment the max of current run
                    currentMax = currentMax + 1;
                % if the zone has changed
                else
                    obj.timeInEachCluster{oldZone,1} = ...
                        [obj.timeInEachCluster{oldZone,1}, currentMax];
                    oldZone = newZone;
                    % Reinitialize current max
                    currentMax = 1;
                end  
            end
        end % end of ExtractTimeInEachCluster function
        
        %% Find the distance of a point from the mean of the clusters
        % Inputs:
        % - x = datapoint
        % Outputs:
        % - d = distances from each cluster
        function [d] = FindDistancesOfPointFromClustersMean(obj, x)
            % Competion and Ranking
            innovation     = x - obj.nodesMean;
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
        
        %% Eliminate the empty clusters
        function obj = EliminateEmptyClusters(obj)
            % Number of nodes found empty until now
            numberOfAlreadyEmptiedNodes = 0;
           
            % Check that all datanodes are full
            for i = 1:size(obj.dataNodesPerClusterStand, 2)
               if isempty(obj.dataNodesPerClusterStand{1, i})
                   
                   % Temporary vector on which to work
                   clusterAssignmentsNew = obj.clusterAssignments;
                   
                   % Move the elements in the vector
                   clusterAssignmentsNew(obj.clusterAssignments >= i - numberOfAlreadyEmptiedNodes) = ...
                       obj.clusterAssignments(obj.clusterAssignments >= i - numberOfAlreadyEmptiedNodes)-1;
                   
                   % Assign temporary value to actual one
                   obj.clusterAssignments  = clusterAssignmentsNew;
                   
                   % Augment counter of eliminated clusters
                   numberOfAlreadyEmptiedNodes = numberOfAlreadyEmptiedNodes + 1;
                   % Also change number of nodes
                   obj.N = obj.N - 1;
               end
            end
        end % end of EliminateEmptyClusters function
        
        %% Function to extract the assignment to the clusters for each 
        % datapoint of the training data.
        function obj = ExtractClusterAssignments(obj)
            % Total length of training data
            dataLength        = size(obj.data,1); 
            
            % Initialize cluster assignments
            obj.clusterAssignments = [];
            
            % Looping over all data to find, for each datapoint, 
            % the closest cluster
            for c = 1:dataLength    
                
                % Take current data point(standardized version, which is 
                % the one used to cluster)
                xStand = obj.dataStand(c,:);
                
                % Find closest cluster to standardized data point
                [closestClusterIndex] = obj.FindClosestClusterGivenStandardizedValues(xStand); 
                
                % Give to cluster assignment
                obj.clusterAssignments = [obj.clusterAssignments; closestClusterIndex];
            end
        end % end of ExtractClusterAssignments function
        
        %% Function to extract the clustering features, if cluster assignment
        % has already been calculated
        function obj = ExtractClusteringFeaturesGivenClusterAssignments(obj)

            % Find beginning of trajectories
            obj = obj.FindBeginningInstantOfTrajectories();

            % Extract overall training data mean
            obj = obj.ExtractClusterMean();
            % Extract overall training data standard deviation
            obj = obj.ExtractClusterStd();

            % Extract the data belonging to each cluster
            obj = obj.ExtractDataNodesPerCluster();
            % Eliminate clusters that are empty
            obj = obj.EliminateEmptyClusters();
            
            % Extract the data belonging to each cluster (redo, in case
            % some clusters had been eliminated).
            obj = obj.ExtractDataNodesPerCluster();
            % Extract the mean values of each cluster
            obj = obj.ExtractClustersMean();
            % Extract the covariance values of each cluster
            obj = obj.ExtractClustersCov();
            % Extract transition matrix
            obj = obj.ExtractTransitionMatrix();
            % Extract max time overall spent in clusters
            obj = obj.ExtractMaxOverallTimeBeforeTransition();
            % Extract temporal transition matrices
            obj = obj.ExtractTemporalTransitionMatrices();
            % Extract max time spent IN EACH cluster
            obj = obj.ExtractMaxClustersTime();
            % Extract time in each cluster
            obj = obj.ExtractTimeInEachCluster();
        end
        
        %% Function to extract the clustering features
        function obj = ExtractClusteringFeatures(obj)
            
            % Extract vector containing cluster assignments
            obj = obj.ExtractClusterAssignments();
            % Extract all other features
            obj = obj.ExtractClusteringFeaturesGivenClusterAssignments();
            
        end % end of ExtractClusteringFeatures function
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Functions for training the Graph from given data
        
        %% Initialize an empty graph
        % Inputs:
        % - inputData
        function [obj, inputData] = InitializeEmptyGraph(obj, inputData)
            % Length and state dimension of data
            dataLength      = size(obj.data,1);     
            stateDimension  = size(obj.data,2);     
            % Generate a random seed
            seed            = RandStream('mt19937ar','Seed', ...
                                         obj.clusteringParameters.seedvector);
            % Use the seed to perform pseudo-random permutation of the data
            inputData       = inputData(randperm(seed,dataLength), :); 
            
            % Min and max value of data
            dataMin = min(inputData);
            dataMax = max(inputData);

            % Initial 2 nodes for training the algorithm
            Ni = 2;    

            obj.nodesMeanStand = zeros(Ni, stateDimension);

            for i = 1:Ni
                % It returns an array of random numbers generated from the continuous 
                % uniform distributions with lower and upper endpoints specified by 
                % 'dataMin' and 'dataMax'.
                obj.nodesMeanStand(i,:) = unifrnd(dataMin, dataMax);    
            end

            obj.E       = zeros(Ni,1);
            obj.utility = ones(Ni,1);
            obj.C       = zeros(Ni, Ni);
            obj.t       = zeros(Ni, Ni);
        end % end of InitializeEmptyGraph function
        
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
        
        %% Function to perform Aging
        % Inputs:
        % - t: age matrix
        % - s1: closest cluster to datapoint
        % Outputs:
        % - t: age matrix
        function [obj, t] = PerformAging(obj, t, s1)
            % Increment the age of all edges emanating from s1
            t(s1, :) = t(s1, :) + 1;                                 
            t(:, s1) = t(:, s1) + 1;
        end % end of PerformAging function
        
        %% Function to perform Adaptation
        % Inputs:
        % - x: datapoint
        % - s1: closest cluster to datapoint
        % - C: connection matrix
        % Outputs:
        % - C: connection matrix
        function [obj] = PerformAdaptation(obj, x, s1)
            % Move the nearest distance node and it's neibors to wards 
            % input signal by fractions Eb and En resp.
            obj.nodesMeanStand(s1,:) = obj.nodesMeanStand(s1,:) + ...
                                       obj.clusteringParameters.epsilon_b*(x-obj.nodesMeanStand(s1,:));   
            %   Take all the connections of the closest node to the data in question
            Ns1 = find(obj.C(s1,:) == 1);         
            
            % Move the direct topological neibors of nearest distance 
            % node (S1) and it's neibors to wards input signal by fractions 
            % Eb and En resp.
            for j = Ns1
                
                obj.nodesMeanStand(j,:) = obj.nodesMeanStand(j,:) + ...
                                          obj.clusteringParameters.epsilon_n*(x-obj.nodesMeanStand(j,:));                         
            end
        end % end of PerformAdaptation function
        
        %% Function to change parameters after having creatde/removed links
        function [obj] = PerformCreationRemovalLinks(...
                                        obj, s1, s2)
            % Create Link
            % If s1 and s2 are connected by an edge , 
            % set the age of this edge to zero , 
            % it such edge doesn't exist create it
            obj.C(s1,s2) = 1;                                                       
            obj.C(s2,s1) = 1;
            obj.t(s1,s2) = 0;   
            % Age of the edge
            obj.t(s2,s1) = 0;
            % Remove Old Links
            % Remove edges with an age larger than Amax(a threshold value)
            obj.C(obj.t > obj.clusteringParameters.T) = 0;                                                       
            nNeighbor = sum(obj.C);      
            % Number of connections of each node
            AloneNodes = (nNeighbor==0);
            obj.C(AloneNodes, :) = [];
            obj.C(:, AloneNodes) = [];
            obj.t(AloneNodes, :) = [];
            obj.t(:, AloneNodes) = [];
            obj.nodesMeanStand(AloneNodes, :) = [];
            obj.E(AloneNodes) = [];
            obj.utility(AloneNodes) = [];
        end % end of PerformCreationRemovalLinks function
        
        %% Function to add a node
        function [obj] = PerformNodeAdding(obj) 
            % Determine the unit q with the maximum accumulated error
            [~, q] = max(obj.E);         
            % Maximum index related to the error related to a connected node
            [~, f] = max(obj.C(:,q).*obj.E);                                        

            % Total number of nodes
            r = size(obj.nodesMeanStand,1) + 1;     
            % Insert a new unit r halfway between q and it's 
            % neighbor f with the largest error variable
            obj.nodesMeanStand(r,:) = (obj.nodesMeanStand(q,:) + obj.nodesMeanStand(f,:))/2;                                   

            %   Remove old connections and introduce the presence of the
            %   new created node
            obj.C(q,f) = 0;
            obj.C(f,q) = 0;
            obj.C(q,r) = 1;
            obj.C(r,q) = 1;
            obj.C(r,f) = 1;
            obj.C(f,r) = 1;
            obj.t(r,:) = 0;
            obj.t(:, r) = 0;

            % Decrease the error variable of q and f by 
            % multiplying them with a constand 'alpha'
            obj.E(q) = obj.clusteringParameters.alpha*obj.E(q);                                              
            obj.E(f) = obj.clusteringParameters.alpha*obj.E(f);
            % Initialize the error of the new node equal to 
            % error of the winner node
            obj.E(r) =   obj.E(q) ;                                    
            % Decrease the error variable of q and f by 
            % multiplying them with a constand 'alpha'
            obj.utility(r) =  0.5 *( obj.utility(q) + obj.utility(f) );
            
            % Plus one cluster
            obj.N = obj.N + 1; 
        end % end of PerformNodeAdding function
        
        %% Function to remove a node
        function [obj] = PerformNodeRemoval(obj) 
            % Maximum accumelated error
            [max_E, ~] = max(obj.E);       
            % Node node_useless having minimum utility
            [min_utility,node_useless] = min(obj.utility);  
            % Utility factor
            CONST = min_utility * obj.clusteringParameters.k;                                        

            if (CONST < max_E)
                % Remove the connection having smaller utility factor
                obj.C(node_useless,:) = [];                                                                          
                obj.C(:,node_useless) = [];
                obj.nodesMeanStand(node_useless,:) = [];   
                % Remove the min utility value from the utility vector
                obj.utility(node_useless) = [];    
                % Remove error vector correspond to the node having min utility
                obj.E(node_useless) = [];    
                % Remove aging vector correspond to the node having min utility
                obj.t(node_useless,:) = [];                                     
                obj.t(:, node_useless) = [];
                
                % Less one cluster
                obj.N = obj.N - 1;
            end            
        end % end of PerformNodeRemoval function
        
        %% Function to find standardized version of data
        function obj = findDataStandardized(obj)
            % Extract overall training data mean
            obj = obj.ExtractClusterMean();
            % Extract overall training data standard deviation
            obj = obj.ExtractClusterStd();
            
            % Find standardized version of the data
            obj.dataStand = obj.data - obj.X_mean;
            obj.dataStand = obj.dataStand./repmat(obj.X_std, size(obj.dataStand, 1), 1); 
        end % end of findDataStandardized function
        
        %% Performing Clustering with Kmeans
        function obj = PerformKmeansClustering(obj, parametersHolder, dataHolder, ...
                                               clusteringParameters, ...
                                               weightFeatures)
                                           
           % Pass parameters holder
           obj.parametersHolder = parametersHolder;
            
           % Input data from parameters holder
           inputData = parametersHolder.dataAndParametersInSingleArray;
                                           
           % Saving data from input to class
           obj.clusteringParameters = clusteringParameters;
           obj.N                    = clusteringParameters.N;
           obj.dataHolder           = dataHolder;
           obj.data                 = inputData;
           obj                      = findDataStandardized(obj);
           obj.weightFeatures       = weightFeatures;

           % Length and state dimension of data
           dataLength      = size(obj.data,1);        
           % Generate a random seed
           seed            = RandStream('mt19937ar','Seed', ...
                                        obj.clusteringParameters.seedvector);
           % Use the seed to perform pseudo-random permutation of the data
           %inputData       = inputData(randperm(seed,dataLength), :); 
           
           % Take only those values with weight > 0
           inputDataConsidered = obj.dataStand(:, obj.weightFeatures ~=0);
           % K-means
           obj.clusterAssignments = kmeans(inputDataConsidered,obj.N);
           
           % Extract the features related to the clusters
           obj = ExtractClusteringFeaturesGivenClusterAssignments(obj);
                                           
        end

        %% Function to create the array of beginning instant of trajectories
        % (necessary to later create a transition matrix that does not 
        % have erroneous jumps)
        function obj = FindBeginningInstantOfTrajectories(obj)

            % How many trajectories does the data contain
            numberOfTrajectories       = size(obj.dataHolder.Data, 1);
            % Array with the starting points of the trajectories
            startingPoints = ones(numberOfTrajectories,1);

            % Sum for keeping track of beginnings
            currentBeginning = 1;

            % Searching for the starting points
            for i = 2:numberOfTrajectories
                lenthOfPreviousTrajectory = size(obj.dataHolder.Data{i-1,1},1);
                currentBeginning = currentBeginning + lenthOfPreviousTrajectory;
                startingPoints(i) = currentBeginning;
            end

            obj.trajectoriesStartingPoints = startingPoints;
        end

        %% Find features from data and already found nodesMeanStand
        function obj = ExtractClusteringFeaturesGivenDataAndNodesMeanStand(obj, inputData, ...
                                            dataHolder, clusteringParameters, ...
                                            weightFeatures, nodesMeanStand, numberOfClusters)

            % Saving data from input to class
            obj.clusteringParameters = clusteringParameters;
            obj.dataHolder           = dataHolder;
            obj.data                 = inputData;
            obj                      = findDataStandardized(obj);
            obj.weightFeatures       = weightFeatures;
            obj.N                    = numberOfClusters;

            % Give assignments
            obj.nodesMeanStand       = nodesMeanStand;

            % Extract the features related to the clusters
            obj = ExtractClusteringFeatures(obj);

        end

        %% Find features from data and already found cluster assignments
        function obj = ExtractClusteringFeaturesGivenDataAndClusterAssignments(obj, inputData, ...
                                            dataHolder, clusteringParameters, ...
                                            weightFeatures, clusterAssignments)

            % Saving data from input to class
            obj.clusteringParameters = clusteringParameters;
            obj.dataHolder           = dataHolder;
            obj.data                 = inputData;
            obj                      = findDataStandardized(obj);
            obj.weightFeatures       = weightFeatures;

            % Give assignments
            obj.clusterAssignments   = clusterAssignments;
            obj.N                    = max(clusterAssignments) - min(clusterAssignments) + 1;

            % Extract the features related to the clusters
            obj = ExtractClusteringFeaturesGivenClusterAssignments(obj);

        end
        
        function [obj, nx] = PerformSingleGNGIteration(obj, dataForClustering, nx)
            
            currentIterationErrors = 0;
            dataLength            = size(obj.data,1); 

            for c = 1:dataLength 

                % Number of nodes
                obj.N = size(obj.nodesMeanStand, 1);

                % Number of data
                % Select Input
                % Counter of cycles inside the algorithm
                nx = nx + 1;      
                % Pick first input vector from permuted inputs
                x  = dataForClustering(c,:);                                                        

                % Competition and Ranking
                [s1, s2, d] = obj.PerformRanking(x);

                % Aging
                [obj, obj.t] = obj.PerformAging(obj.t, s1);

                % Add Error
                dist0 = d(s1)^2;
                dist1 = d(s2)^2;
                obj.E(s1) = obj.E(s1) + dist0;

                currentIterationErrors = currentIterationErrors + dist0;

                % Utility
                % Initial utility is zero in first case and dist is 
                % the error of first node
                deltaUtility =  dist1 - dist0;     
                % Difference between error of two nodes
                obj.utility(s1)  =  obj.utility(s1) + deltaUtility ;                         

                % Adaptation
                [obj] = obj.PerformAdaptation(x, s1);

                % Add and remove links
                [obj] = obj.PerformCreationRemovalLinks(s1, s2);

                % Add New Nodes
                if mod(nx, obj.clusteringParameters.L_growing) == 0 && ...
                   size(obj.nodesMeanStand,1) < obj.clusteringParameters.N
                    [obj] = obj.PerformNodeAdding();
                end

                % Remove Node
                if mod(nx, obj.clusteringParameters.L_decay) == 0 ...
                        && size(obj.nodesMeanStand,1) > 2
                    [obj] = obj.PerformNodeRemoval();
                end 

                % Decrease Errors
                % Decrease error variables by multiplying them with a constant delta
                obj.E = obj.clusteringParameters.delta * obj.E;             
                % Decrease the utility by alpha_utility constant
                obj.utility = obj.clusteringParameters.delta * obj.utility;                                           
            end 

            obj.meanErrorsInTraining = [obj.meanErrorsInTraining, currentIterationErrors];
            obj.numberOfNeuronsInTraining = [obj.numberOfNeuronsInTraining, obj.N];
            
        end
        
        %% Performing Clustering with GNG
        function obj = PerformGNGClustering(obj, inputData, dataHolder, ...
                                            clusteringParameters, ...
                                            weightFeatures)
                    
            % Saving data from input to class
            obj.clusteringParameters = clusteringParameters;
            obj.dataHolder           = dataHolder;
            obj.data                 = inputData;
            obj                      = findDataStandardized(obj);
            obj.weightFeatures       = weightFeatures;
            obj.meanErrorsInTraining = [];
            obj.numberOfNeuronsInTraining = [];

            % Initialization
            [obj, dataForClustering] = InitializeEmptyGraph(obj, obj.dataStand);
            
            % Perform GNG clustering
            nx = 0;
            totalNumberIterations = obj.clusteringParameters.MaxIt;
            
            
            for it = 1:totalNumberIterations
                
                % Print what is the current iteration
                disp(it);
                
                [obj, nx] = obj.PerformSingleGNGIteration(dataForClustering, nx);
                
            end  % end of iteration loop
            
            
            % Extract the features related to the clusters
            obj = ExtractClusteringFeatures(obj);
        end % end of PerformGNGClustering function
          
    end % end of methods

end