
function [params] = Config_clustering()

%% GNG parameters

% Iteration (repetition of input data)
params.MaxIt = 50; 
% Growing rate
params.L_growing = 1500;
% Movement of winner node
params.epsilon_b = 0.1; 
% Movement of all other nodes except winner
params.epsilon_n = 0.0006;
% Decaying global error and utility
params.alpha = 0.5;              
% Decaying global error and utility
params.delta = 0.9995;     
% Decaying local error and utility
params.T = 20;    
% Decay rate. 
% It could be a function of params.L_growing, e.g., params.LDecay = 2*params.L_growing
% Decay rate sould be faster than the growing then it will remove extra nodes
params.L_decay = 1000;                                                       
params.alpha_utility = 0.0005;  
% Seed for randomization
params.seedvector = 50;

% Number of max clusters
params.N                   = 90;
% K parameter for stopping the GNG clusters creation
params.k                   = 70;%10;

%% Weights of each state/params part of odometry 
% The elements are, from left to right:
% rotation angle, position along x, position along y, 
% acceleration along x, acceleration along y,
% velocity along x, velocity along y, norm of velocity
% These are not used for filtering, but just to extract the desired
% outputs after filtering.
% They will be more relevant in clustering.
params.weights             = [0 1 1 0 0 1 1 0];

%% Orthogonal GNG?
% GNG with orthogonal component given more importance?
params.orthogonalGNG = true;
% Weights to give in the case of using the GNG with orthogonal
% importance.
% Elements:
% perpendicular distance, parallel distance, velocity innovation along x
% velocity innovation along y
params.weightsForNetworkOrthogonal = [10, 1, 1, 1];

%% VAE states?
% Clustering also using the states of the VAE?
params.considerVAEStates = false;
if params.orthogonalGNG == true
    params.considerVAEStates = false;
end

end