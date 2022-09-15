
% Class to perform the graph matching between a source graph and a 
% set of target graphs in order to extract the clusters specific to 
% the source graph and its distances.

classdef MultipleGraphComparer
    
    properties
        
        % Source Graph
        sourceGraph
        % Target Graphs
        % This is a list of target graphs
        targetGraphs
        % This is a list of graph matchers
        graphMatchers
        % Number of target graphs
        numberOfTargetGraphs
        
    end
    
    methods
        
        %% Constructor
        % Inputs - varargin
        % - first argument is the source graph
        % - the other arguments are the target graphs
        function obj = MultipleGraphComparer(varargin)
            % > The first input is the SOURCE Graph
            obj.sourceGraph      = varargin{1};
            
            % > The other inputs are the TARGET Graphs and they are 
            %   inserted in a list
            %   List length = number of inputs minus the source one
            obj.numberOfTargetGraphs = nargin - 1;
            %   List creationg
            obj.targetGraphs = [];
            for i = 1 : obj.numberOfTargetGraphs
                singleTargetGraph = varargin{i+1};
                obj.targetGraphs  = [obj.targetGraphs; singleTargetGraph];
            end
            
            % > Creation of graph matchers
            %   List of Graph Matchers
            obj.graphMatchers = [];
            for i = 1 : obj.numberOfTargetGraphs
                % Create current Graph Matcher
                singleGraphMatcher = ...
                    GraphMatcher(obj.sourceGraph, obj.targetGraphs(i));
                % Insert in the list of Graph Matchers
                obj.graphMatchers  = [obj.graphMatchers; singleGraphMatcher];
            end
        end % end of constructor
        
        %% Function to find the mean distances for clusters of source
        %  graph over all target graphs
        function averagedMinimumDistances = FindMeanClusterDistancesAsSimpleEuclidean(obj, varargin)
            for i = 1: obj.numberOfTargetGraphs
                % Find the minimum distances over clusters for current 
                % graph matcher, so related to couple of source graph and 
                % a current target graph.
                minimumDistancesCurrentmatcher = ...
                    obj.graphMatchers(i).FindClusterDistancesAsSimpleEuclidean(varargin{1});
                % If it was the first target, inizialize the sum with the distances
                if i == 1
                    sumOfMinimumDistances = minimumDistancesCurrentmatcher;
                % Otherwise, add the value to the sum
                else
                    sumOfMinimumDistances = sumOfMinimumDistances + minimumDistancesCurrentmatcher;
                end
            end  
            % Average value over Graph matchers, so over target graphs
            averagedMinimumDistances = sumOfMinimumDistances/obj.numberOfTargetGraphs;
        end % end of averagedMinimumDistances function
        % VERSION with standardized data and cluster centers
        function averagedMinimumDistances = FindMeanClusterDistancesAsSimpleEuclideanStand(obj, varargin)
            for i = 1: obj.numberOfTargetGraphs
                % Find the minimum distances over clusters for current 
                % graph matcher, so related to couple of source graph and 
                % a current target graph.
                minimumDistancesCurrentmatcher = ...
                    obj.graphMatchers(i).FindClusterDistancesAsSimpleEuclideanStand(varargin{1});
                % If it was the first target, inizialize the sum with the distances
                if i == 1
                    sumOfMinimumDistances = minimumDistancesCurrentmatcher;
                % Otherwise, add the value to the sum
                else
                    sumOfMinimumDistances = sumOfMinimumDistances + minimumDistancesCurrentmatcher;
                end
            end  
            % Average value over Graph matchers, so over target graphs
            averagedMinimumDistances = sumOfMinimumDistances/obj.numberOfTargetGraphs;
        end % end of averagedMinimumDistances function
        
    end
    
end