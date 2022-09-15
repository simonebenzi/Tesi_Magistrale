
function [] = Alg103_o_PlotTrajectories()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adding code paths and defining data paths
addpath('./MATLAB_paths');
paths = DefineCodeAndDataPaths();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% For defining figure ranges
hugeNumber = 10000000000;
minX = hugeNumber;
minY = hugeNumber;
maxX = -hugeNumber;
maxY = -hugeNumber;
offset = 0.005; %this is a ratio w.r.t. the overall range
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Video resolution
resolution = 1200;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Figure definition
TrainingOdometry = figure;
TrainingOdometry.Position = [0,0,600,600];
hold on
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Printing the odometry
% 0 = training, 1 = validation, 2 = testing
for dataCase = 0:2
    % Name of input and output based on the data case
    if dataCase == 0
        path_to_positions_cells      = paths.path_to_training_positions_cells;
        color = 'r';
    elseif dataCase ==1
        path_to_positions_cells      = paths.path_to_validation_positions_cells;
        color = 'g';
    elseif dataCase == 2
        path_to_positions_cells      = paths.path_to_test_positions_cells;
        color = 'b';
    end    
    %% Load and plot
    [positions, isLoaded] = loadObjectGivenFileName(path_to_positions_cells);
    if isLoaded == true
        
        for i = 1:size(positions,2)
    
            currPositions = positions{1,i};
            s = scatter(currPositions(:, 1), currPositions(:,2), 2, ...
                color, 'filled');
            q = quiver(currPositions(1:end-1, 1), currPositions(1:end-1,2), ...
                currPositions(2:end, 1) - currPositions(1:end-1, 1), ...
                currPositions(2:end, 2) - currPositions(1:end-1, 2), ...
                color, 'Autoscale', 'off');

            if i > 1
                s.Annotation.LegendInformation.IconDisplayStyle = 'off';
                q.Annotation.LegendInformation.IconDisplayStyle = 'off';
            end

            currMinX = min(currPositions(:,1));
            currMaxX = max(currPositions(:,1));
            currMinY = min(currPositions(:,2));
            currMaxY = max(currPositions(:,2));

            if currMinX < minX
               minX = currMinX;
            end
            if currMaxX > maxX
               maxX = currMaxX;
            end
            if currMinY < minY
               minY = currMinY;
            end
            if currMaxY > maxY
               maxY = currMaxY;
            end
        end       
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adding legends to figure
grid on
xlabel('x', 'FontSize',11, 'Interpreter','latex')
ylabel('y', 'FontSize',11, 'Interpreter','latex')
legend({'Training positions', 'Training speeds', 'Validation positions', 'Validation speeds', ...
    'Testing positions', 'Testing speeds'},  ...
    'FontSize',9, 'Interpreter','latex', ...
    'Location', 'northoutside', ...
    'Orientation', 'vertical')
rangeX = maxX - minX;
rangeY = maxY - minY;
axis([minX - rangeX*offset, maxX + rangeX*offset, ...
      minY - rangeY*offset, maxY + rangeY*offset])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save
ax = gca;
% Requires R2020a or later
exportgraphics(ax, [paths.path_saved_odometry_plot '.png'],'Resolution',resolution) 

end
