
% CLASS for loading and holding the data for training/testing
% This class reads the data from a file and inserts it in a cell array.
% Additionally, if necessary, it adds a sinusoid to the data.
% In the case in which the sinusoid is not to add (because it is already
% implicitly embedded in it or it is not necessary), the variable
% 'Data' and 'DataPlusSin' will have the same content.

classdef DataHolder
    

    %% Properties
    properties
        % Contains the data for training or testing
        Data 
        % Contains the data with an embedded sinusoid
        % If no sinusoid is embedded in the data, 
        % this variable has the same content as 'Data'
        DataPlusSin 
    end
    
    %% Private properties
    properties (Access = private)
        
    end
    
    %% Methods
    methods
        
        %% Constructor
        % inputs:
        % If nargin == 1:
        % - caseData: says which data must be taken by defining it with a 
        %   code.
        % If nargin == 2 (pass the data directly)
        % - data
        % - dataPlusSin (can be = data if there is no sinusoid to add)
        function obj = DataHolder(varargin)            
            if nargin == 1
                caseData           = varargin{1};
                % Loads the data structure
                dataStructure      = DataHolder.LoadStructure(caseData);

                % Puts the data in a cell array called 'data'
                fns                = fieldnames(dataStructure);
                data               = dataStructure.(fns{1});

                % Saves the data to Data and DataPlusSin
                obj.Data           = data;
                obj.DataPlusSin    = data;
            elseif nargin == 2
                obj.Data           = varargin{1};
                obj.DataPlusSin    = varargin{2};
            end       
        end % end constructor
        
        %% Set function for Data property
        function obj = set.Data(obj, newData)
            obj.Data = newData;
        end % end set function for Data
        
        %% Set function for DataPlusSin property
        function obj = set.DataPlusSin(obj, newData)
            obj.DataPlusSin = newData;
        end % end set function for DataPlusSin

        %% Excludes cells from the data
        % This function eliminates one of the data cells
        % Inputs:
        % - excludedCell: index of the cell to exclude
        function obj = ExcludeDataCells(obj, excludedCell)
            % How many cells the data contains
            numDataCells      = size(obj.Data, 1);
            % How many cells to eliminate
            numCellsToExclude = length(excludedCell);
            
            % Initialize empty cell space for less data 
            newData = cell(numDataCells - numCellsToExclude, 1);
            
            % Take out the excluded trajectories
            count = 1;
            for i = 1: numDataCells
                % If this is the index of cell to exclude
                if sum(excludedCell == i) > 0
                else
                    newData{count, 1} = obj.Data{i, 1};
                    count = count + 1;
                end
            end
            
            obj.Data = newData;
        end % end of ExcludeDataCells function
        
        %% Take only first cell of data
        % Function to eliminate all the data cells except the first one
        function obj = TakeOnlyFirstCell(obj)
            % Single cell
            newData      = cell(1,1);
            % Take only first cell
            newData{1,1} = obj.Data{1,1};
            
            % Substitute
            obj.Data = newData;   
        end
        
        %% Adds a sinusoid to the data
        % Function to add a sinusoid to the data
        % Inputs are the information about the sinusoid:
        % - range of sinusoid
        % - period of sinusoid
        % - phase of sinusoid
        % - offset of sinusoid
        function obj = AddSinusoid(obj, range, period, phase, offset)
            % Take base data
            data_plus_sin = obj.Data;
            % How many cells does the data contain?
            numDataCells      = size(obj.Data, 1);
            
            % Loop over the number of cells of the data
            for curr_cell = 1:numDataCells

                cell_pos_plus_sin = [];

                % Extracting position data
                cell_pos          = obj.Data{curr_cell, 1};
                cell_data_len     = size(cell_pos, 1);
                % Extracting velocity data
                cell_vel          = cell_pos;
                for i = 2:cell_data_len
                   cell_vel(i, :) = cell_pos(i,:) - cell_pos(i-1, :); 
                end
                % Normalized_vel
                cell_vel_norm     = cell_vel/(norm(cell_vel));

                x                 = 1:1:cell_data_len;
                cell_sinusoid     = range*sin(2*pi*(x)/period + phase) + offset;
                
                % Looping over all the time instants
                for i = 1:size(cell_vel, 1)

                    curr_pos      = cell_pos(i, :);
                    curr_vel_norm = cell_vel_norm(i, :);
                    curr_vel_orth = null(curr_vel_norm);

                    curr_vel_sin  = cell_sinusoid(i);

                    cell_pos_plus_sin(i, :) = curr_pos + curr_vel_sin*curr_vel_orth';

                    if i > 2

                        curr_vel_plus_sin = cell_pos_plus_sin(i, :) - prev_pos_plus_sin;

                        % new vel should be directed along same verse of velocity
                        % Project velocity on it
                        projection_vel_curr_on_past_vel = ...
                            GeometryHandler.ProjectPointOnPlane3D(prev_vel_plus_sin, curr_vel_plus_sin);
                        % Find sign based if same sign or inverse
                        sign_multiplied    = curr_vel_plus_sin.*(projection_vel_curr_on_past_vel);

                        on_other_side = norm(cell_pos_plus_sin(i, :)- prev_pos_plus_sin) > 0.75*range;
                        if sum(sum(sign_multiplied> 0))>1 && on_other_side == false
                            sign = 1;
                        else
                            sign = -1;
                        end

                        cell_pos_plus_sin(i, :) = curr_pos + sign*curr_vel_sin*curr_vel_orth';

                    end
                    if i > 1
                        prev_vel_plus_sin   = cell_pos_plus_sin(i, :) - prev_pos_plus_sin;
                    end
                    prev_pos_plus_sin       = cell_pos_plus_sin(i, :);
                end
                data_plus_sin{curr_cell, 1} = cell_pos_plus_sin;
            end
            
            obj.DataPlusSin = data_plus_sin;
        end
        
        %% PLOTTING FUNCTIONS
        
        %% Plots the positional data of a chosen Cell
        % input:
        % - chosenCell: cell index of the data to plot
        % SCATTER VERSION
        function Scatterplot2DDataSingleCell(obj, chosenCell)
            figure
            obj.ScatterSingleCell(chosenCell);
            xlabel('x')
            ylabel('y')
            legend({'positional data'})
            title(['Positional data in cell ', num2str(chosenCell)])  
        end % end Scatterplot2DDataSingleCell function
        % PLOT VERSION
        function Plot2DDataSingleCell(obj, chosenCell)
            figure
            obj.PlotSingleCell(chosenCell);
            xlabel('x')
            ylabel('y')
            legend({'positional data'})
            title(['Positional data in cell ', num2str(chosenCell)])
            shg  
        end % end Scatterplot2DDataSingleCellPlusSin function
        % PLOT VERSION with TIME
        function Plot2DDataSingleCellInTime(obj, chosenCell)
            figure
            obj.PlotSingleCellInTime(chosenCell);
            xlabel('x')
            ylabel('y')
            zlabel('time')
            legend({'positional data in time'})
            title(['Positional data in cell ', num2str(chosenCell)])
            shg  
        end % end Plot2DDataSingleCellInTime function

        %% Plots the positional data of a chosen Cell, plus Sinusoid
        % input:
        % - chosenCell: cell index of the data to plot
        % SCATTER VERSION
        function Scatterplot2DDataSingleCellPlusSin(obj, chosenCell)
            figure
            obj.ScatterSingleCellPlusSin(chosenCell);
            xlabel('x')
            ylabel('y')
            legend({'positional data'})
            title(['Positional data plus sin in cell ', num2str(chosenCell)])
            shg  
        end % end Scatterplot2DDataSingleCellPlusSin function
        % PLOT VERSION
        function Plot2DDataSingleCellPlusSin(obj, chosenCell)
            figure
            obj.PlotSingleCellPlusSin(chosenCell);
            xlabel('x')
            ylabel('y')
            legend({'positional data'})
            title(['Positional data plus sin in cell ', num2str(chosenCell)])
            shg  
        end % end Scatterplot2DDataSingleCellPlusSin function
        
        %% Scatters the positional data of a chosen Cell
        % input:
        % - chosenCell: cell index of the data to plot
        function ScatterSingleCell(obj, chosenCell)
            scatter(obj.Data{chosenCell, 1}(:, 1), ...
                    obj.Data{chosenCell, 1}(:, 2), 'r');
        end % end ScatterSingleCell function
        %% Scatters the positional data of a chosen Cell, with sinusoid
        % input:
        % - chosenCell: cell index of the data to plot
        function ScatterSingleCellPlusSin(obj, chosenCell)
            scatter(obj.DataPlusSin{chosenCell, 1}(:, 1), ...
                    obj.DataPlusSin{chosenCell, 1}(:, 2), 'r');
        end % end ScatterSingleCell function
        %% Plots the positional data of a chosen Cell, with sinusoid
        % input:
        % - chosenCell: cell index of the data to plot
        function PlotSingleCellPlusSin(obj, chosenCell)
            plot(obj.DataPlusSin{chosenCell, 1}(:, 1), ...
                 obj.DataPlusSin{chosenCell, 1}(:, 2), 'r');
        end % end ScatterSingleCell function
        %% Plots the positional data of a chosen Cell
        % input:
        % - chosenCell: cell index of the data to plot
        function PlotSingleCell(obj, chosenCell)
            plot(obj.Data{chosenCell, 1}(:, 1), ...
                 obj.Data{chosenCell, 1}(:, 2), 'r');
        end % end ScatterSingleCell function
        % This version does it also plotting time on the 3rd axis
        function PlotSingleCellInTime(obj, chosenCell)
            plot3(obj.Data{chosenCell, 1}(:, 1), ...
                  obj.Data{chosenCell, 1}(:, 2), ...
                  1:1:size(obj.Data{chosenCell, 1}, 1), ...
                  'r');
        end % end ScatterSingleCell function
        
    end % end methods
    
    methods (Static)
        
        %% Function to load a data structure based on data index given
        % Inputs:
        % - caseData: says which data must be taken by defining it with a 
        %   code.
        function dataStructure = LoadStructure(caseData)
            % Take data structure based on index
            if caseData     == 0
                dataStructure    = load('datacell_F_noOvertake_lowVorticity_3trajs.mat');
                %structure    = load('datacell_F_noOvertake_lowVorticity_3trajs_sin.mat');
            elseif caseData == 1
                dataStructure    = load('datacell_F_noOvertake_highVorticity.mat');
            elseif caseData == 2
                dataStructure    = load('datacell_F_Overtake_lowVorticity.mat');
                %structure    = load('datacell_F_Overtake_lowVorticity_sin.mat');
            elseif caseData == 3
                dataStructure    = load('datacell_F_Overtake_highVorticity.mat');
            elseif caseData == 4
                dataStructure    = load('straight_lowV_cells_trajs_smoothed.mat');
            elseif caseData == 5
                dataStructure    = load('straight_highV_cells_trajs_smoothed.mat');
            elseif caseData == 6
                dataStructure    = load('change_lowV_cells_trajs_smoothed.mat');
            elseif caseData == 7
                dataStructure    = load('change_highV_cells_trajs_smoothed.mat');
            elseif caseData == 8
                dataStructure    = load('DataPMCell.mat');
            elseif caseData == 100
                dataStructure    = load('DataOACell.mat');
            elseif caseData == 101
                dataStructure    = load('DataPUTurnCell.mat');
            elseif caseData == 102
                dataStructure    = load('DataESCell.mat');
            elseif caseData == 9
                dataStructure    = load('DatacellsMotorwayNormal.mat');
            elseif caseData == 10
                dataStructure    = load('DatacellsMotorwayDrowsy.mat');
            elseif caseData == 11
                dataStructure    = load('DatacellsMotorwayAggressive.mat');
            elseif caseData == 12
                dataStructure    = load('DatacellsSecondaryNormal.mat');
            elseif caseData == 13
                dataStructure    = load('DatacellsSecondaryDrowsy.mat');
            elseif caseData == 14
                dataStructure    = load('DatacellsSecondaryAggressive.mat');
            elseif caseData == 15
                dataStructure    = load('DatacellsMotorwayNormalAugmented.mat');
            elseif caseData == 16
                dataStructure    = load('DatacellsMotorwayDrowsyAugmented.mat');
            elseif caseData == 17
                dataStructure    = load('DatacellsMotorwayAggressiveAugmented.mat');
            elseif caseData == 25
                % Params of circle/line creation
                lenLine = 500;
                ratioLengthLineCircle = 2;
                radiusCircle = 10;
                % Create circle/line
                DataCreator.CreateCircleLine1(lenLine, ...
                                              ratioLengthLineCircle, ...
                                              radiusCircle);                        
                % Load created Circle/Line
                dataStructure    = load('CircleLineData1.mat'); 
            elseif caseData == 26
                % Params of circle/line creation
                lenLine = 500;
                ratioLengthLineCircle = 2;
                radiusCircle = 10;
                velFactorSecondLine = 2;
                % Create circle/line
                DataCreator.CreateCircleLine2(lenLine, ...
                                              ratioLengthLineCircle, ...
                                              radiusCircle, ...
                                              velFactorSecondLine);                      
                % Load created Circle/Line
                dataStructure    = load('CircleLineData2.mat');
            end
        end % end loadStructure function
        
    end % end of static methods
    
end