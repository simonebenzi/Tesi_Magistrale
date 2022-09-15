

classdef ClusteringGraphOrthogonal < ClusteringGraph
    
    properties
        orthogonals
        doubleOrthogonals
        
        predictionsBase
        predictionsBaseCells
        
        predictionErrors
        predictionErrorsCells
        
        orthogonalPredictionErrors
        orthogonalPredictionErrorsCells
    end
    
    
    methods
        
        %% Empty constructor
        function obj = ClusteringGraphOrthogonal()
            % Call ClusteringGraph constructor
            obj@ClusteringGraph();
        end % end of constructor
        
        %% Learn orthogonal graph from base graph
        % Inputs of different possible length:
        % If 2 inputs:
        % - base Graph
        % - parameters for GNG clustering
        % Weights set in an equal way
        % If 3 inputs:
        % - base Graph
        % - parameters for GNG clustering
        % - weights for features (orthogonal error and its derivative)
        function obj = LearnOrthogonalGraphFromBaseGraph(obj, varargin)
            
            % Base Graph
            baseGraph                = varargin{1};
            % Parameters used in clustering
            obj.clusteringParameters = varargin{2};

            % Extract the orthogonal parameters, predictions and errors
            obj            = obj.ExtractOrthogonalErrorsFromBaseGraph(baseGraph);
            % Find the derivative of orthogonal error
            obj            = obj.FindOrthogonalErrorDerivative();
            % Create the dataHolder
            obj.dataHolder = DataHolder(obj.data, obj.data);
            
            % Find weighting values
            if nargin == 3
                dimensionOrthogonalErrorState = size(obj.data, 2);
                % Weights to give to the different features - > same for all
                obj.weightFeatures = ones(1, dimensionOrthogonalErrorState)/...
                                          dimensionOrthogonalErrorState;
            elseif nargin == 4
                % Weights to give to the different features
                obj.weightFeatures = varargin{3};
            end
            
            % Perform the clustering
            obj            = obj.PerformGNGClustering(obj.data, obj.dataHolder, ...
                                                      obj.clusteringParameters, ...
                                                      obj.weightFeatures);
        end % end of LearnOrthogonalGraphFromBaseGraph function
        
        %% Function to find the derivative of orthogonal error
        % and insert it into the data
        function obj = FindOrthogonalErrorDerivative(obj)
            % Initialize with prediction errors
            orthogonalPredictionErrorsDerivative = obj.orthogonalPredictionErrors;
            % Loop to calculate derivative
            for i = 2:size(obj.orthogonalPredictionErrors, 1)
                orthogonalPredictionErrorsDerivative(i) = obj.orthogonalPredictionErrors(i) - ...
                                                          obj.orthogonalPredictionErrors(i-1);
            end
            orthogonalPredictionErrorsDerivative(1) = 0;
            
            % Data parameter on which to cluster will be the 
            obj.data = [obj.orthogonalPredictionErrors, ...
                        orthogonalPredictionErrorsDerivative];
        end % end of FindOrthogonalErrorDerivative function
        
        %% Function to initialize the different elements necessary for orthogonal
        % graph calculation
        function obj = InitializeOrthogonalsPredictionsAndErrors(obj, dataCellNumbers)
            % Orthogonals
            obj.orthogonals                     = [];
            obj.doubleOrthogonals               = [];
            % Predictions
            obj.predictionsBase                 = [];
            obj.predictionsBaseCells            = cell(dataCellNumbers, 1);
            % Prediction errors
            obj.predictionErrors                = [];
            obj.predictionErrorsCells           = cell(dataCellNumbers, 1);
            % Prediction errors along orthogonal
            obj.orthogonalPredictionErrors      = [];
            obj.orthogonalPredictionErrorsCells = cell(dataCellNumbers, 1);  
        end % end of InitializeOrthogonalsPredictionsAndErrors function
        
        %% Find the errors along the orthogonal from the baseGraph
        % Inputs:
        % - baseGraph: this is the base clustering on which prediction
        %   to find the error
        function obj = ExtractOrthogonalErrorsFromBaseGraph(obj, baseGraph)
            % How many data cells in the data
            dataCellNumbers = size(baseGraph.dataHolder.Data, 1);
            % Initialize to null orthogonals, predictions and errors
            obj             = obj.InitializeOrthogonalsPredictionsAndErrors(dataCellNumbers);
            
            % Loop over the number of cells in the data
            for n = 1:dataCellNumbers
                
                % Sequence of cluster assignations for current Cell
                currentCellClusterAssignments   = baseGraph.clusterAssignmentsCells{n, 1};
                % Data of the current Cell, BEFORE FILTERING
                currentCellDataUnfiltered       = baseGraph.dataHolder.DataPlusSin{n, 1};
                % Data of the current Cell, AFTER FILTERING
                currentCellDataFilteredPosition = baseGraph.parametersHolder.filteredData{n, 1};
                currentCellDataFilteredVelocity = baseGraph.parametersHolder.velocities{n, 1};
                
                % First prediction errors to zero
                stateDimension                  = size(currentCellDataFilteredPosition, 2);
                obj.predictionErrors            = [obj.predictionErrors, ...
                                                  zeros(1, stateDimension)];
                obj.orthogonalPredictionErrors  = [obj.orthogonalPredictionErrors; ...
                                                  zeros(1,stateDimension/2)];
                obj.predictionsBase             = [obj.predictionsBase, ...
                                                  zeros(1, stateDimension)];
                                              
                % Looping over the size of the current data
                for i = 1: size(currentCellDataUnfiltered, 1)-1

                    % Positions and velocities of current cluster
                    currPositionData     = currentCellDataFilteredPosition(i,:);
                    currVelocityData     = currentCellDataFilteredVelocity(i,:);
                    nextPositionData     = currentCellDataUnfiltered(i+1, :);
                    % Current cluster assignment
                    currCluster          = currentCellClusterAssignments(i);

                    %% Find orthogonal to line from center to current point
                    % Find connection between center of circle and current data point
                    lineCenterPoint      = currPositionData - ...
                                             baseGraph.rotationCenters(currCluster,:);
                    % Normalize the connection
                    lineCenterPointNorm = sqrt(lineCenterPoint(1).^2 + lineCenterPoint(2).^2);
                    lineCenterPointNorm = lineCenterPoint/(lineCenterPointNorm +1e-50);

                    % Find orthogonal to line_center_point towards mean direction of motion
                    if sum(lineCenterPointNorm == 0 ) == stateDimension
                        orthogonal      = zeros(1, stateDimension);
                    else
                        % Find orthogonal to line_center_point towards mean direction of motion
                        orthogonal      = null(lineCenterPointNorm);
                    end

                    % orthogonal should be directed along same verse of velocity
                    % Project velocity on it
                    projectionVelocityOnOrthogonal = ...
                        GeometryHandler.ProjectPointOnPlane3D(currVelocityData, orthogonal');
                    % Find sign based if same sign or inverse
                    signMultiplied    = projectionVelocityOnOrthogonal.*(orthogonal');
                    if sum(sum(signMultiplied> 0))>1
                        sign = 1;
                    else
                        sign = -1;
                    end
                    orthogonal = orthogonal*sign;
                    
                    %% Rotate of 90 degrees
                    R = [0 -1; 1 0];
                    doubleOrthogonal = R*(orthogonal);

                    %% Predicted velocity and position along direction to attractor
                    % Predicted velocity
                    predictedVelocity(:, i) = orthogonal*baseGraph.meanNormVel(currCluster);
                    % Predicted position
                    predictedPosition(:, i) = currPositionData + predictedVelocity(:, i)';

                    %% Distance between real point and predicted one
                    distancePredictionReal     = nextPositionData' - predictedPosition(:, i);
                    % Norm of the distance
                    distancePredictionRealNorm = norm(distancePredictionReal);

                    %% Sign 
                    %  So checking projection on double_orthogonal
                    projectionDistanceOnOrthogonal = ...
                        GeometryHandler.ProjectPointOnPlane3D(distancePredictionReal', doubleOrthogonal');
                    % Find sign based if same sign or inverse
                    signMultiplied = projectionDistanceOnOrthogonal.*(doubleOrthogonal');
                    if sum(sum(signMultiplied> 0))>1
                        sign = -1;
                    else
                        sign = 1;
                    end

                    %% ERROR
                    orthogonalPositionError          = distancePredictionRealNorm*sign;

                    %% Save everything
                    % Orthogonals
                    obj.orthogonals                  = [obj.orthogonals orthogonal];
                    obj.doubleOrthogonals            = [obj.doubleOrthogonals, doubleOrthogonal];
                    % Predictions
                    obj.predictionsBase              = [obj.predictionsBase; ...
                                                        predictedPosition(:,i)'];
                    % Errors
                    obj.predictionErrors             = [obj.predictionErrors; ...
                                                        predictedPosition(:,i)' - nextPositionData];
                    obj.orthogonalPredictionErrors   = [obj.orthogonalPredictionErrors; ...
                                                        orthogonalPositionError];   
                end
                
                %% Save Cells
                obj.predictionsBaseCells{n, 1}   = obj.predictionsBase;
                obj.predictionErrorsCells{n,1}   = obj.predictionErrors;
                obj.orthogonalPredictionErrorsCells{n, 1} = obj.orthogonalPredictionErrors;
            end 
        end % end of ExtractOrthogonalErrorsFromBaseGraph function  
        
        %% Function to smoothen the orthogonal prediction errors
        function obj = SmoothenOrthogonalPredictionErrors(obj)
            % Smooth the overall
            obj.orthogonalPredictionErrors           = smooth(obj.orthogonalPredictionErrors);
            % Smooth over the cells
            for i = 1:size(obj.orthogonalPredictionErrors, 1)
                obj.orthogonalPredictionErrors{n, 1} = smooth(obj.orthogonalPredictionErrors{n, 1});
            end
        end 
        
        %% PLOTTING FUNCTIONS
        %% Function to plot prediction errors
        function PlotPredictionErrors(obj)
            figure
            plot(obj.predictionErrors)
            xlabel('time')
            ylabel('error')
            title('Prediction errors')
        end % end of PlotPredictionErrors function
        %% Function to plot orthogonalprediction errors
        function PlotOrthogonalPredictionErrors(obj)
            figure
            plot(obj.orthogonalPredictionErrors)
            xlabel('time')
            ylabel('error')
            title('Orthogonal prediction errors')
        end % end of PlotPredictionErrors function
        %% Function to plot both orthogonal error and its derivative, 
        %  once the clustering is performed
        function PlotOrthogonalErrorAndDerivative1D(obj)
            figure
            subplot(2, 1, 1)
            plot(obj.data(:, 1))
            xlabel('time')
            ylabel('orthogonal error')
            title('orthogonal errors')
            subplot(2, 1, 2)
            plot(obj.data(:, 1))
            xlabel('time')
            ylabel('orthogonal error derivative')
            title('orthogonal error derivatives')
        end % end of PlotOrthogonalErrorAndDerivative1D function
        %% Function to plot the predicted position with the rotation centers
        function PlotPredictedPositionAndRotationCenters(obj, baseGraph)
            figure
            
        end
           
    end    
end