
%% After the separate odometry points from the Carla simulator
% have been extracted, this code combines them together in a single
% matlab cell array.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adding code paths and defining data paths
addpath('./MATLAB_paths');
paths = DefineCodeAndDataPaths();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Select the files related to the positional data
filePattern = fullfile(paths.path_to_positions_folder, 'real_position_train*.*mat');
fileList    = dir(filePattern);
fileArray   = struct2cell(fileList);
% Create a cell array where to put all the positional data
datacell = cell(1, size(fileArray, 2));
%% Loop over the trajectories, to insert the data
for i = 1: size(fileArray, 2)
    
    i
    
    % Current name
    name = fileArray(1, i);
    name = name{1,1};
    % Current data
    dataCurr = load([paths.path_to_positions_folder, name]);
    dataCurr = squeeze(dataCurr.positions);
    [r, c] = size(dataCurr);
    if c > 1
        dataCurr = dataCurr(:, 1:2);
    else
        dataCurr = dataCurr(1:2);
    end
    % Insert in cell array
    datacell{1, i} = dataCurr;
    
end
%% Save
position = datacell;
save(paths.path_to_positions_cells, 'position');
