
% CLASS for incrementing the holding of data by also calculating and 
% holding the parameters of motion.

classdef ParametersHolder
    

    %% Properties
    properties
        
        % This defines what is the length of the memory we want
        % to have to mean the past
        memoryLength
        % This weights the past 'memoryLength' vectors
        memoryWeights
        
        % Data after filtering
        filteredData
        
        % Velocities
        velocities
        velocitiesNorm
        % Thetas
        thetas
        % Accelerations
        txtys
        
        % Data and parameters in cells
        dataAndParameters
        % Data and parameters in a single array
        dataAndParametersInSingleArray
      
    end
    
    %% Private properties
    properties (Access = private)
        
    end
    
    %% Methods
    methods
        
        %% Constructor
        function obj = ParametersHolder(dataHolder, filter, memoryLength, memoryWeights)
            
            obj.memoryLength  = memoryLength;
            obj.memoryWeights = memoryWeights;
            
            obj = obj.ExtractMotionParameters(dataHolder);
            obj = obj.PerformFilteringOverAllDataCells(dataHolder, filter);
            obj = obj.CombineAllDataAndParametersInCells();
            obj = obj.CombineAllDataAndParameters();
            
        end % end constructor
        
        %% PLOTTING FUNCTIONS
        function PlotAllDataAndParameters2DFirstCell(obj)
            
            figure 
            
            % Position
            subplot(5, 1, 1)
            scatter(obj.filteredData{1,1}(:,1), obj.filteredData{1,1}(:,2))
            xlabel('x')
            ylabel('y')
            title('Positions of first data cell')
            % Velocity
            subplot(5, 1, 2)
            scatter(obj.velocities{1,1}(:,1), obj.velocities{1,1}(:,2))
            xlabel('vx')
            ylabel('vy')
            title('Velocities of first data cell')
            % Theta
            subplot(5, 1, 3)
            plot(obj.thetas{1,1}(50:end))
            xlabel('theta')
            title('Rotation angles of first data cell')
            % Acceleration
            subplot(5, 1, 4)
            hold on
            plot(obj.txtys{1,1}(50:end, 1), 'r')
            plot(obj.txtys{1,1}(50:end, 2), 'b')
            xlabel('Acc')
            legend({'tx', 'ty'})
            title('Accelerations of first data cell')
            % Velocity norm
            subplot(5, 1, 5)
            plot(obj.velocitiesNorm{1,1}(50:end, 1), 'r')
            xlabel('norm velocity')
            legend({'vel norm'})
            title('Norm velocities of first data cell')
            
            shg
        end
        
        %% FILTERING FUNCTIONS
        
        %% Function to perform filtering over all the data Cells
        function obj = PerformFilteringOverAllDataCells(obj, dataHolder, filter)
            
            % Of how many cells is composed the data we are given
            numberOfDataCells = size(dataHolder.DataPlusSin, 1);
            
            % Initialize filtered data with original data
            obj.filteredData      = dataHolder.DataPlusSin;
            
            % Looping over all the data cells
            for i = 1 : numberOfDataCells
                
                % Extracting the data of the current cell
                dataOfCurrentCell = dataHolder.DataPlusSin{i, 1};
                velocitiesOfCurrentCell = obj.velocities{i, 1};
                
                % Applying the filter on the data
                [filteredDataCurrentCell] = ...
                    filter.PerformKalmanFilteringWithVelocityGiven( ...
                    dataOfCurrentCell', velocitiesOfCurrentCell'); 
                filteredDataCurrentCell   = filteredDataCurrentCell';
                
                % Put filtered data in the cell
                obj.filteredData{i, 1}        = filteredDataCurrentCell;
                
            end 
        end % end of PerformFilteringOverAllDataCells function
        
        %% Function to extract the motion parameters
        function obj = ExtractMotionParameters(obj, dataHolder)
            
            % Of how many cells is composed the data we are given
            numberOfDataCells = size(dataHolder.DataPlusSin, 1);
            
            % Initialize the velocities, thetas and txtys
            obj.velocities     = cell(numberOfDataCells, 1);
            obj.velocitiesNorm = cell(numberOfDataCells, 1);
            obj.thetas         = cell(numberOfDataCells, 1);
            obj.txtys          = cell(numberOfDataCells, 1);
            
            % Looping over all the data cells
            for i = 1 : numberOfDataCells
                
                % States of current cell
                positionsOfCurrentCell  = dataHolder.DataPlusSin{i, 1};
                
                % Motion parameters to extract from current cell
                velocitiesOfCurrentCell     = [];
                velocitiesNormOfCurrentCell = [];
                thetasOfCurrentCell         = [];
                txtysOfCurrentCell          = [];
                
                % For the first time instant, value to zero
                velocitiesOfCurrentCell     = [velocitiesOfCurrentCell;
                   ParametersHolder.DefineFirstVelocity(positionsOfCurrentCell)];
                velocitiesNormOfCurrentCell = [velocitiesNormOfCurrentCell;
                   ParametersHolder.DefineFirstVelocityNorm()];
                thetasOfCurrentCell         = [thetasOfCurrentCell;
                   ParametersHolder.DefineFirstTheta()];
                txtysOfCurrentCell          = [txtysOfCurrentCell;
                   ParametersHolder.DefineFirstTxTy()];

                % Looping over the data in a single cell
                for j = 2 : size(dataHolder.DataPlusSin{i, 1}, 1)
                    
                    % Block of positions
                    positionsInMemoryLength = ...
                        obj.SelectMemoryLengthBlockOfPositions( ...
                        positionsOfCurrentCell, j);
                    
                    % Previous velocity
                    previousVelocity = velocitiesOfCurrentCell(j-1, :);
                    
                    % Find the motion parameters
                    currentVelocity  = ...
                        obj.FindVelocityValueAtGivenTimeWithMemory(...
                        positionsInMemoryLength);
                    currentVelocityNorm  = ...
                        obj.FindVelocityNormValueAtGivenTimeWithMemory(...
                        currentVelocity);
                    [currentTheta, currentTxTy] = ...
                        ParametersHolder.FindThetaTxTyValueAtGivenTime( ...
                        previousVelocity, currentVelocity);
                    
                    % Insert the parameters in the array
                    velocitiesOfCurrentCell = [velocitiesOfCurrentCell; currentVelocity];
                    velocitiesNormOfCurrentCell = ...
                        [velocitiesNormOfCurrentCell; currentVelocityNorm];
                    thetasOfCurrentCell     = [thetasOfCurrentCell; currentTheta];
                    txtysOfCurrentCell      = [txtysOfCurrentCell; currentTxTy];
  
                end
                
                % This is to polish the theta definition for areas
                % where velocity is very low, so we might have weird
                % oscillations.
                thetasOfCurrentCell = obj.PolishThetaDefinition(...
                        thetasOfCurrentCell, velocitiesNormOfCurrentCell);
                
                % Save final values
                obj.velocities{i, 1}     = velocitiesOfCurrentCell;
                obj.velocitiesNorm{i, 1} = velocitiesNormOfCurrentCell;
                obj.thetas{i, 1}         = thetasOfCurrentCell;
                obj.txtys{i, 1}          = txtysOfCurrentCell;
            end
        end % end of ExtractMotionParameters function
        
        %% Function to combine all the data and parameters in an array of cells
        function obj = CombineAllDataAndParametersInCells(obj)
            
            % Of how many cells is composed the data we are given
            numberOfDataCells = size(obj.filteredData, 1);
            
            % Initialize data and parameters array
            obj.dataAndParameters = cell(numberOfDataCells, 1);
            
            % Looping over all the data cells
            for i = 1 : numberOfDataCells
                
                % Looping over the data in a single cell
                for j = 1 : size(obj.filteredData{i, 1}, 1)
                    
                    % Positions, velocities, thetas and accelerations
                    % of current data point
                    currentData = [obj.thetas{i, 1}(j, :), ...
                                   obj.filteredData{i, 1}(j, :), ...
                                   obj.txtys{i, 1}(j, :), ...
                                   obj.velocities{i, 1}(j, :), ...
                                   obj.velocitiesNorm{i, 1}(j, :)];
                               
                    % Add them to the vector
                    obj.dataAndParameters{i, 1} = [ ...
                        obj.dataAndParameters{i, 1}; currentData];
                end
            end
        end % end of CombineAllDataAndParameters function
        
        %% Function to combine all the data and parameters in a single
        %  array
        function obj = CombineAllDataAndParameters(obj)
            
            % Of how many cells is composed the data we are given
            numberOfDataCells = size(obj.filteredData, 1);
            
            % Initialize data and parameters array
            obj.dataAndParametersInSingleArray = [];
            
            % Looping over all the data cells
            for i = 1 : numberOfDataCells
                
                % Looping over the data in a single cell
                for j = 1 : size(obj.filteredData{i, 1}, 1)
                    
                    % Positions, velocities, thetas and accelerations
                    % of current data point
                    currentData = [obj.thetas{i, 1}(j, :), ...
                                   obj.filteredData{i, 1}(j, :), ...
                                   obj.txtys{i, 1}(j, :), ...
                                   obj.velocities{i, 1}(j, :), ...
                                   obj.velocitiesNorm{i, 1}(j, :)];
                               
                    % Add them to the vector
                    obj.dataAndParametersInSingleArray = [ ...
                        obj.dataAndParametersInSingleArray; currentData];
                end
            end
        end % end of CombineAllDataAndParameters function
        
        %% Function to select positions in a certain memory length space
        function positions = SelectMemoryLengthBlockOfPositions( ...
                obj, positionsCell, index)
            
            beginIndex = max(1, index - obj.memoryLength);
            endindex   = index;
            
            positions  = positionsCell(beginIndex:endindex, :);
            
        end % end of SelectMemoryLengthBlockOfPositions function
        
        %% Function to find the value of the velocity at a certain time by
        %  averaging over the past time instants
        function currentVelocity = FindVelocityValueAtGivenTimeWithMemory(...
                        obj, positionsInMemoryLength)
                    
            % Velocities from positions
            velocitiesInMemoryLength = diff(positionsInMemoryLength);
            
            % Cut the memory weights, in case these are the first time
            % instants of the sequence
            lengthCut        = size(velocitiesInMemoryLength, 1);
            memoryWeightsCut = obj.memoryWeights(1:lengthCut);
            % normalize
            memoryWeightsCut = memoryWeightsCut/sum(memoryWeightsCut); 
            
            % Averaging the velocities
            currentVelocity  = memoryWeightsCut*velocitiesInMemoryLength;
                    
        end % end of FindVelocityValueAtGivenTimeWithMemory function 
        
        %% Function to make the theta definition better
        function thetasOfCurrentCell = PolishThetaDefinition( obj, ...
                        thetasOfCurrentCell, velocitiesNormOfCurrentCell)
                    
            % Setting place for new thetas
            newThetasOfCurrentCell = thetasOfCurrentCell;
            
            % Mean and standard value of norm velocity
            meanVelocityNorm = mean(velocitiesNormOfCurrentCell);
            stdVelocityNorm  = std(velocitiesNormOfCurrentCell);

            % Total time instants
            numTimeInstants = size(thetasOfCurrentCell, 1);
            
            % Looping over the number of time instants
            for i = 1: numTimeInstants
                
                % Select begin and end instant of window
                beginTime = max(1, i - 2*ceil(obj.memoryLength));
                endTime   = min(numTimeInstants, i + 2*ceil(obj.memoryLength));
                % Take the thetas in that window
                thetasInZone = thetasOfCurrentCell(beginTime:endTime);
                
                % Velocity norm in time instant
                velNormCurr  = velocitiesNormOfCurrentCell(i);
                
                % Zero crossing detector
                zcd = dsp.ZeroCrossingDetector;
                % How many zero crossings
                numZeroCross = zcd(thetasInZone);
                
                % Conditions:
                % - are there at least 2 zero crossings?
                % - is velocity norm low?
                atLeast2ZeroCrossings = numZeroCross >= 2;
                veryLowSpeed          = ...
                    velNormCurr < meanVelocityNorm - 2*stdVelocityNorm;
                
                if atLeast2ZeroCrossings && veryLowSpeed
                    newThetasOfCurrentCell(beginTime:endTime) = 0 + 1e-20;
                end 
            end
            
            thetasOfCurrentCell = newThetasOfCurrentCell;
        end % end of PolishThetaDefinition function
        
    end % end methods
    
    methods (Static)
        
        %% Function to find the value of the velocity norm at a certain time
        function currentVelocityNorm = FindVelocityNormValueAtGivenTimeWithMemory(...
                currentVelocity)
            currentVelocityNorm = norm(currentVelocity);
        end
        
        %% Function to define the first value of velocity in a cell
        function firstVelocity = DefineFirstVelocity(positions)
            
            firstVelocity = zeros(1, size(positions, 2));
            
        end % end of firstVelocity function
        
        %% Function to define the first value of velocity norm in a cell
        function firstVelocity = DefineFirstVelocityNorm()
            
            firstVelocity = 0;
            
        end % end of firstVelocity function
        
        %% Function to define the first value of theta in a cell
        function firstTheta = DefineFirstTheta()
            
            firstTheta = 0;
            
        end % end of defineFirstTheta function
        
        %% Function to define the first value of accelerations in a cell
        function firstTxTy = DefineFirstTxTy()
            
            firstTxTy = [0 0];
            
        end % end of defineFirstTxTy function
        
        %% Function to find the value of velocity at a certain time
        function currentVelocity = FindVelocityValueAtGivenTime( ...
                previousPosition, currentPosition)
            
            currentVelocity = currentPosition - previousPosition;
            
        end % end of FindVelocityValueAtTimei function 
        
        %% Function to find the value of theta at a certain time
        function [currentTheta, currentTxTy] = FindThetaTxTyValueAtGivenTime( ...
                previousVelocity, currentVelocity)
            
            % Normalize them
            currentVelocityNorm    = sqrt(currentVelocity(1)^2  + currentVelocity(2)^2);
            previousVelocityNorm   = sqrt(previousVelocity(1)^2 + previousVelocity(2)^2);
            
            % Extract versors of velocities
            if currentVelocityNorm ~= 0
                currentVelocityVersor = currentVelocity/currentVelocityNorm;
            else
                currentVelocityVersor = currentVelocity;
            end
            if previousVelocityNorm ~= 0
                previousVelocityVersor = previousVelocity/previousVelocityNorm;
            else
                previousVelocityVersor = previousVelocity;
            end
            
            % Find rotation angle
            sin_A = (currentVelocityVersor(2)*previousVelocityVersor(1) - ...
                     currentVelocityVersor(1)*previousVelocityVersor(2));
            cos_A = (currentVelocityVersor(1) + ...
                     sin_A*previousVelocityVersor(2))/previousVelocityVersor(1);
                 
            %% Theta 
            theta = asin(sin_A);
            thetaCheck   = acos(cos_A);
            
            % Theta cases
            if     theta >= 0 && thetaCheck >= 0 
                different = sum((thetaCheck < (theta - abs(theta)*0.001))) || ...
                    sum((thetaCheck > (theta + abs(theta)*0.001)));  
            elseif (theta < 0 && thetaCheck > 0) || (theta > 0 && thetaCheck < 0)  
                different = sum((thetaCheck < (-theta - abs(theta)*0.001))) || ...
                    sum((thetaCheck > (-theta + abs(theta)*0.001))) ;
            elseif isnan(theta)
                different = 1;
                theta = 0;
            else
                different = 1;
                theta = thetaCheck;
            end
            
            currentTheta = real(theta);
            
            %% TxTy
            
            % Take txty as the remaining error after using theta
            if different == 1 || theta == 0
                tx = currentVelocity(1) - previousVelocity(1);
                ty = currentVelocity(2) - previousVelocity(2);
            else
                tx = currentVelocity(1) - ...
                    (cos_A*previousVelocity(1) - sin_A*previousVelocity(2));
                ty = currentVelocity(2) - ...
                    (sin_A*previousVelocity(1) + cos_A*previousVelocity(2));
            end
            
            currentTxTy = [tx ty];
            
        end % end of FindThetaValueAtTimei function 
    end
end