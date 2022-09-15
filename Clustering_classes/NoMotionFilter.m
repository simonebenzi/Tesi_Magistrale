
% Function for no motion filter: this is a Kalman Filter that supposes
% that no motion is taking place

classdef NoMotionFilter < KalmanFilter 
    
    properties
        
    end
    
    methods
        
        %% Function for creating transition matrix (Abstract in Kalman
        % Filter).
        % It is an Identity matrix (keeps same position)
        % Inputs:
        % - stateDimension
        function obj = CreateTransitionMatrix(obj, stateDimension)
            obj.transitionMatrix = eye(stateDimension);
        end % end of CreateTransitionMatrix
        
        %% Function for creating observation matrix (Abstract in Kalman
        % Filter).
        % It is an Identity matrix (takes observation)
        % Inputs:
        % - observationDimension
        function obj = CreateObservationMatrix(obj, observationDimension)
            obj.observationMatrix = eye(observationDimension);
        end % end of CreateObservationMatrix
        
        %% CONSTRUCTOR
        % Inputs:
        % - observationVariance
        % - predictionVariance
        % - initialVariance
        % - observationDimension
        % - stateDimension
        function obj = NoMotionFilter(observationVariance, predictionVariance, initialVariance, ...
                                       observationDimension, stateDimension)
            % Call KalmanFilter constructor
            obj@KalmanFilter(observationVariance, predictionVariance, initialVariance, ...
                             observationDimension, stateDimension);
            
            obj = obj.CreateTransitionMatrix(stateDimension);
            obj = obj.CreateObservationMatrix(observationDimension);
        end % end of constructor
        
    end % end of methods
end