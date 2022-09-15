
function [] = Alg104_ov_MakeVideoOfTrajectory(frame_rate,max_number_of_trajectories)

% Default values, if some input is not provided
if nargin<2
    % Number of trajectories to plot
    max_number_of_trajectories = 1;
end
if nargin<1
    % Frame rate of video
    frame_rate    = 30;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adding code paths and defining data paths
addpath('./MATLAB_paths');
paths = DefineCodeAndDataPaths();
% List of the image folders
path_to_training_images_folder = paths.path_to_training_images_folder;
names_of_image_data_folders = dir(path_to_training_images_folder);
% How to name the video
name_of_video = paths.path_to_video_of_images_vs_odometry;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load odometry data
[positions, isLoaded] = loadObjectGivenFileName(paths.path_to_training_positions_cells_norm);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Make video
outputVideo = VideoWriter(name_of_video,'MPEG-4');
outputVideo.FrameRate = frame_rate; % <----------- FRAME RATE here
open(outputVideo)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Loop for video creation
figDB2 = figure;
figDB2.Position = [300 100 800 400];
number_of_trajectories = min(max_number_of_trajectories, size(positions, 2));
for i = 1:number_of_trajectories

    iFolderName = names_of_image_data_folders(i+2).name;
    workingDir_curr_trajectory = fullfile(path_to_training_images_folder, ...
                                  iFolderName);
 
    imageNames = [dir(fullfile(workingDir_curr_trajectory,'*.jpg')); ...
                  dir(fullfile(workingDir_curr_trajectory,'*.png'))];
    imageNames = {imageNames.name}';

    currentTrajectoryPositions = positions{1,i};

    for k = 1: size(imageNames, 1)-1

        img  = imread(fullfile(workingDir_curr_trajectory,imageNames{k}));

        subplot(1,2,1)
        cla
        image(img);
        colormap(gca,gray(256));
        set(gca,'xtick',[])
        set(gca,'ytick',[])

        subplot(1,2,2)
        cla
        scatter(currentTrajectoryPositions(:,1), currentTrajectoryPositions(:,2), 'b')
        hold on
        scatter(currentTrajectoryPositions(k,1), ...
                currentTrajectoryPositions(k,2), ...
                'r', 'filled')
        xlabel('x')
        ylabel('y')

        frame = getframe(gcf);
        writeVideo(outputVideo,frame)
    end
end
close(outputVideo)

end
