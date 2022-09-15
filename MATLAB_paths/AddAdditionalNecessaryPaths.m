
function [] = AddAdditionalNecessaryPaths()

% Simple utils for handling folders
addpath('./Utils_folders_handling');

% Adding the path to the clustering functions
addpath('./Clustering_classes');
addpath('./ekfukf');

% Where the GNG parameters and other parameters are defined
addpath('./ConfigurationFiles');
addpath('./Utils_clustering');

% Utils for debugging the final localization
addpath('./Utils_localization_debugging');

end
