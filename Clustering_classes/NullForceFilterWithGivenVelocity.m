
% Function for null force filter: this is a Kalman Filter that supposes
% that we move with same velocity as before.
% In this case, instead of taking the velocity as difference between 
% observation at time k and estimation at k-1, it is supposed to be given.
% It is used for both prediction and update.

classdef NullForceFilterWithGivenVelocity < NullForceFilter
    
    properties
        
    end
    
    methods
        
        %% CONSTRUCTOR
        % Inputs:
        % - observationVariance
        % - predictionVariance
        % - initialVariance
        % - observationDimension
        % - stateDimension
        function obj = NullForceFilterWithGivenVelocity(observationVariance, ...
                predictionVariance, initialVariance, observationDimension, stateDimension)
            % Call NullForceFilter constructor
            obj@NullForceFilter(observationVariance, predictionVariance, initialVariance, ...
                             observationDimension, stateDimension);   
        end
        
        %% Function for perform Kalman Filter
        % Inputs:
        % - observations
        % - velocities
        function [estimatedStates] = PerformKalmanFilteringWithVelocityGiven( ...
                obj, observations, velocities)
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
                    % Take velocity
                    estimatedStateCurrent(3:4) = velocities(:, ii-1);
                    % Perform prediction
                    [estimatedStateCurrent, estimatedCovarianceCurrent] = ...
                     performKFPrediction(obj, estimatedStateCurrent, estimatedCovarianceCurrent);
                end
                % UPDATE
                if ii > 1
                    % Take velocity
                    currentObservedValueVel =  velocities(:, ii);
                    currentObservedValue    = [observations(:, ii) ; currentObservedValueVel];
                end
                % Perform update
                [estimatedStateCurrent, estimatedCovarianceCurrent] = ...
                performKFUpdate(obj, estimatedStateCurrent, estimatedCovarianceCurrent, ...
                                currentObservedValue);
                % Add estimated state
                estimatedStates(:, ii) = estimatedStateCurrent(1:size(observations, 1));  
            end
        end % end of PerformKalmanFiltering function
        
    end % end of methods   
end