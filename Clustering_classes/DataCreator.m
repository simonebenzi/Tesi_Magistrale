
% CLASS for creating very simple new data for training and testing
% to be used for dummy examples while coding.

classdef DataCreator
    
    %% Methods
    methods (Static)
        
        %% Function to build a circle
        % input:
        % - lenCircle: length of Circle
        % - radiusCircle: radius of Circle
        % output:
        % - circleX: circle positions along x
        % - circleY: circle positions along y
        function [circleX, circleY] = CreateCircle(lenCircle, radiusCircle)
            %% Circle building
            % Center in 0
            x       = 0;
            y       = 0;
            % Radius
            r       = radiusCircle;
            % Angle
            th      = 0:pi/lenCircle:2*pi;
            % Circle along x
            circleX = r * cos(th) + x;
            % Circle along y
            circleY = r * sin(th) + y;            
        end % end CreateCircle function
        
        %% Function to extract the final velocity of a structure
        % input:
        % - dataX: all the data position, along x
        % - dataY: all the data position, along y
        % output:
        % - VxLast: final velocity along x
        % - VyLast: final velocity along y
        function [VxLast, VyLast] = ExtractFinalVelocity(dataX, dataY)
            % All velocities
            vx     = dataX(2:end)-dataX(1:end-1);
            vy     = dataY(2:end)-dataY(1:end-1);
            % Last velocity
            VxLast = vx(end/2);
            VyLast = vy(end/2);
        end % end ExctractFinalVelocity function
        
        %% Function to create a line with constant velocity
        % input: 
        % - lenLine: length of line
        % - vx: velocity along x
        % - vy: velocity along y
        % - startPointX: starting x position of line
        % - startPointY: starting y position of line
        % output:
        % - line: contains xy positions of line
        function [line] = CreateLineConstantVelocity(lenLine, vx, vy, ...
                                                     startPointX, startPointY)                                
            % Initialize line                                  
            line       = zeros(lenLine, 2);
            
            % First point is beginning plus velocity
            line(1, :) = [startPointX + vx, ...
                          startPointY + vy];
                      
            % Adding all the other line points based on fixed velocity
            for i = 2: lenLine
               line(i, :) = [line(i-1, 1)+ vx, ...
                             line(i-1, 2)+ vy];
            end
        end
        
        %% Create Circle + Line
        function [] = CreateCircleLine1(lenLine, ...
                                        ratioLengthLineCircle, ...
                                        radiusCircle)                  
            %% Circle
            % Length of circle
            lenCircle          = ratioLengthLineCircle*lenLine;
            
            % Circle data
            [circleX, circleY] = DataCreator.CreateCircle(lenCircle, radiusCircle);
                             
            % Extracting velocity from circle (for adding line after it)
            [VxLast, VyLast]   = DataCreator.ExtractFinalVelocity(circleX, circleY);
            
            %% Line
            % Line building with constant velocity
            [line] = DataCreator.CreateLineConstantVelocity(lenLine, VxLast, VyLast, ...
                                                    circleX(ceil(end/2)), ...
                                                    circleY(ceil(end/2)));
                                            
            %% Combine
            % Combining circle and line
            data       = cell(1, 1);
            % Half Circle
            data{1, 1} = [circleX(1:end/2)', circleY(1:end/2)'];
            % Half Circle + line
            data{1, 1} = [data{1, 1}; line];
            
            %% Saving
            save('CircleLineData1.mat', 'data');
        end % end CreateCircle function
        
        %% Create Circle + Line + LineWithDiffVel
        function [] = CreateCircleLine2(lenLine, ...
                                        ratioLengthLineCircle, ...
                                        radiusCircle, velFactorSecondLine)
            %% Circle
            % Length of circle
            lenCircle          = ratioLengthLineCircle*lenLine;
            
            % Circle data
            [circleX, circleY] = DataCreator.CreateCircle(lenCircle, radiusCircle);
                             
            % Extracting velocity from circle (for adding line after it)
            [VxLast, VyLast]   = DataCreator.ExtractFinalVelocity(circleX, circleY);
            
            %% Line
            % Line building with constant velocity
            [line] = DataCreator.CreateLineConstantVelocity(lenLine, VxLast, VyLast, ...
                                                    circleX(ceil(end/2)), ...
                                                    circleY(ceil(end/2)));
                                                
            %% And then another piece with double velocity
            lineDoubleVel = DataCreator.CreateLineConstantVelocity(lenLine, ...
                                                    VxLast*velFactorSecondLine, ...
                                                    VyLast*velFactorSecondLine, ...
                                                    line(end, 1), ...
                                                    line(end, 2));
                                            
            %% Combine
            % Combining circle and line
            data       = cell(1, 1);
            % Half Circle
            data{1, 1} = [circleX(1:end/2)', circleY(1:end/2)'];
            % Half Circle + lines
            data{1, 1} = [data{1, 1}; line; lineDoubleVel];
            
            %% Saving
            save('CircleLineData2.mat', 'data');
        end % end loadStructure function
         
    end % end methods
    
end