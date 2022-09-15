
%% This is a base CLASS for creating a KALMAN FILTER
% Class for defining base Kalman Filter
% This is an ABSTRACT class: the functions for creating the transition
% matrix and observation matrix are not inserted, to adjust to possible
% different behaviors

classdef KalmanFilter
    
    properties
        transitionMatrix
        observationMatrix
        predictionNoiseCovariance
        observationNoiseCovariance
        initialCovariance
    end
    
    methods (Abstract)
        obj = CreateTransitionMatrix(obj)
        obj = CreateObservationMatrix(obj)
    end
    
    methods
        
        %% Function to define the observation noise covariance matrix
        % Inputs:
        % - observationVariance
        % - observationDimension
        function obj = CreateObservationNoiseCovariance(obj, ...
                observationVariance, observationDimension)
            obj.observationNoiseCovariance = eye(observationDimension)*observationVariance;
        end % end of CreateObservationNoiseCovariance function
        
        %% Function to define the prediction noise covariance matrix
        % Inputs:
        % - predictionVariance
        % - stateDimension
        function obj = CreatePredictionNoiseCovariance(obj, ...
                predictionVariance, stateDimension)
            obj.predictionNoiseCovariance = eye(stateDimension)*predictionVariance; 
        end % end of CreatePredictionNoiseCovariance function
        
        %% Function to define the initial covariance matrix
        % Inputs:
        % - initialVariance
        % - stateDimension
        function obj = CreateInitialCovariance(obj, initialVariance, stateDimension)
            obj.initialCovariance = eye(stateDimension)*initialVariance;  
        end % end of CreateInitialCovariance function
        
        %% Constructor
        % Loading the parameters of the Kalman Filter
        % Inputs:
        % - observationVariance
        % - predictionVariance
        % - initialVariance
        % - observationDimension
        % - stateDimension
        function obj = KalmanFilter(observationVariance, predictionVariance, initialVariance, ...
                                    observationDimension, stateDimension)
           obj = obj.CreateObservationNoiseCovariance(observationVariance, observationDimension);
           obj = obj.CreatePredictionNoiseCovariance(predictionVariance, stateDimension);
           obj = obj.CreateInitialCovariance(initialVariance, stateDimension);
        end % end of constructor
        
        %% Function to perform base Kalman Filtering
        function [estimatedStates] = PerformKalmanFiltering(obj, observations)
            % Initialize the covariance with the initial one
            estimatedCovarianceCurrent = obj.initialCovariance;
            % Initialize the state with the first data point
            estimatedStateCurrent      = observations(:, 1);
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
                % Current bservation
                currentObservedValue = observations(:, ii);
                % Perform Update
                [estimatedStateCurrent, estimatedCovarianceCurrent] = ...
                performKFUpdate(obj, estimatedStateCurrent, estimatedCovarianceCurrent, ...
                                currentObservedValue);
                % Add estimated state
                estimatedStates(:, ii) = estimatedStateCurrent;
            end
        end % end of PerformKalmanFiltering function
        
        %% Function to perform prediction
        % Inputs:
        % - estimatedStatePrevious
        % - estimatedCovariancePrevious
        function [estimatedStateCurrent, estimatedCovarianceCurrent] = ...
                performKFPrediction(obj, estimatedStatePrevious, estimatedCovariancePrevious)
            %--------------------- 1) PREDICTION -----------------------------   
            
            % Estimated state at k-th time instant = Estimated state at previous time
            % instant projected through the transition matrix.
            estimatedStateCurrent      = ...
                           obj.transitionMatrix*estimatedStatePrevious;
                       
            % Estimated covariance at current time instant.
            estimatedCovarianceCurrent = ...
                           obj.predictionNoiseCovariance + obj.transitionMatrix * ...
                           estimatedCovariancePrevious *(obj.transitionMatrix)';    
        end % end of performKFPrediction fuction
        
        %% Function to perform update
        % Inputs:
        % - estimatedStatePrevious
        % - estimatedCovariancePrevious
        % - observedValue
        function [estimatedStateCurrent, estimatedCovarianceCurrent] = ...
                performKFUpdate(obj, estimatedStatePrevious, estimatedCovariancePrevious, ...
                observedValue)
            %------------------------------ 2) UPDATE --------------------------------

            % We calculate the INNOVATION: it is the difference between 
            % what I predicted and what I observed newly(which is projected through the 
            % observation matrix):
            innovation = observedValue - obj.observationMatrix * estimatedStatePrevious;

            % Covariance of the innovation:
            innovationCovariance = obj.observationMatrix * estimatedCovariancePrevious * ...
                                  (obj.observationMatrix)' + obj.observationNoiseCovariance;

            % KALMAN GAIN: it considers uncertainty in prediction and uncertainty in
            % the sensors and can thus be used to weight the final prediction through
            % the innovation.
            kalmanGain = estimatedCovariancePrevious * ...
                        (obj.observationMatrix)' * inv(innovationCovariance);

            % FINAL PREDICTION UPDATE AT THE CURRENT INSTANT:
            
            % State prediction: we update the prediction done in the PREDICTION stage
            % throught the new observation.
            % As weight we use the Kalman gain: if the Kalman gain is high and I am
            % very confident in the sensorial data I got, I will correct a lot,
            % otherwise little.
            estimatedStateCurrent      = estimatedStatePrevious + kalmanGain * innovation;
            % We also update the covariance estimation:
            estimatedCovarianceCurrent = estimatedCovariancePrevious - ...
                         kalmanGain * innovationCovariance * (kalmanGain)';
            
        end % end of performKFUpdate fuction
    end % end of methods
    
    methods (Static)
        
        % Function to plot original observations vs. filtering
        % Inputs:
        % - observations
        % - estimatedStates (filtered data)
        function Plot2DObservationsVsFiltering(observations, estimatedStates)
            figure
            scatter(observations(1, :), observations(2, :), 'r');
            hold on 
            scatter(estimatedStates(1, :), estimatedStates(2, :), 'b');
            legend({'observations', 'filtered values'})
            title('Observations vs. Filtered values')
            
        end % end of Plot2DObservationsVsFiltering function
    end % end of static methods
end