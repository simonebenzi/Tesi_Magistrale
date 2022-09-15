
classdef ClusteringGeneralizedGraph
    
    properties
        
        % Graph for the direction of motion
        graphBase
        % Graph for the direction perpendicular to motion
        graphOrthogonal

    end
    
    methods
        
        % Constructor
        function obj = ClusteringGeneralizedGraph(graphBase, graphOrthogonal)
            
            obj.graphBase       = graphBase;
            obj.graphOrthogonal = graphOrthogonal;
            
        end
    end
end