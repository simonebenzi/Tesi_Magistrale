
% Function for null force filter: this is a Kalman Filter that supposes
% that we move with same velocity as before

classdef NullForceFilter < KalmanFilter
    
    properties
        
    end
    
    methods
        
        %% Function for creating transition matrix (Abstract in Kalman
        % Filter).
        % It is a matrix that takes the previous position and adds the
        % previous velocity.
        % Inputs:
        % - stateDimension
        function obj = CreateTransitionMatrix(obj, stateDimension)
            obj.transitionMatrix = eye(stateDimension);
            for i = 1: stateDimension/2
                
                % To add the elements shown with the arrow in example 
                % with stateDimension = 4.
                % obj.transitionMatrix = [ 1 0 1 <--    0 ;
                %                          0 1 0        1 <--;
                %                          0 0 1        0 ;
                %                          0 0 0        1 ];
                
                obj.transitionMatrix(i, i + stateDimension/2) = 1;
            end            
        end % end of CreateTransitionMatrix function
        
        %% Function for creating observation matrix (Abstract in Kalman
        % Filter).
        % It is an Identity matrix (takes observation)
        % Inputs:
        % - observationDimension
        function obj = CreateObservationMatrix(obj, observationDimension)
            obj.observationMatrix = eye(observationDimension);
        end
        
        %% CONSTRUCTOR
        % Inputs:
        % - observationVariance
        % - predictionVariance
        % - initialVariance
        % - observationDimension
        % - stateDimension
        function obj = NullForceFilter(observationVariance, predictionVariance, initialVariance, ...
                                       observationDimension, stateDimension)
            % Call KalmanFilter constructor
            obj@KalmanFilter(observationVariance, predictionVariance, initialVariance, ...
                             observationDimension, stateDimension);
            
            obj = obj.CreateTransitionMatrix(stateDimension);
            obj = obj.CreateObservationMatrix(observationDimension);
        end
        
        %% Function for perform Kalman Filter
        % Inputs:
        % - observations
        function [estimatedStates] = PerformKalmanFiltering(obj, observations)
            % Initialize the covariance with the initial one
            estimatedCovarianceCurrent = obj.initialCovariance;
            % Initialize the state with the first data point
            estimatedStateCurrent      = observations(:, 1);
            estimatedStateCurrent      = [ estimatedStateCurrent; ...
                                           zeros(size(estimatedStateCurrent, 1), 1)];
            currentObservedValue       = estimatedStateCurrent;
            % Initialize estimated states with observations
            estimatedStates            = observations;
            
            % Length of the data to filter
            dataLength                 = size(observations,2);
            
            % Looping over data length
            for ii = 1: dataLength
                % No prediction at first time instant
                if ii ~= 1
                    % PREDICTION
                    [estimatedStateCurrent, estimatedCovarianceCurrent] = ...
                     performKFPrediction(obj, estimatedStateCurrent, estimatedCovarianceCurrent);
                end
                % UPDATE
                if ii > 1
                    % Velocity observed is the current observation, minus
                    % the estimated position at previous time instant
                    currentObservedValueVel =  observations(:, ii) - estimatedStates(:, ii-1);
                    currentObservedValue    = [observations(:, ii) ; currentObservedValueVel];
                end
                % Perform Update
                [estimatedStateCurrent, estimatedCovarianceCurrent] = ...
                performKFUpdate(obj, estimatedStateCurrent, estimatedCovarianceCurrent, ...
                                currentObservedValue);
                % Add estimated state
                estimatedStates(:, ii) = estimatedStateCurrent(1:size(observations, 1));  
            end
        end % end of PerformKalmanFiltering function
        
    end % end of methods
end