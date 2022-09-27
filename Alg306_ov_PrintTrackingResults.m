
close all
clc
clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adding code paths and defining data paths
addpath('./MATLAB_paths');
paths = DefineCodeAndDataPaths();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load IMU
load('C:\Users\simob\Desktop\Old_training\Old_training\02\test_IMU\linearAcceleration.mat')
load('C:\Users\simob\Desktop\Old_training\Old_training\02\test_IMU\orientation.mat')
acc_bias = mean(linearAcceleration(1:100,:));
linearAcceleration = linearAcceleration - acc_bias;% - [0.05, 0];
%% Other necessary code
AddAdditionalNecessaryPaths()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters to define
% Final time instant to print
T = 100; % Starting point for debug
endingPoint = -1;
last_time_to_plot = 1999;
linearAcceleration = linearAcceleration(1:last_time_to_plot,:);
orientationSynch = orientationSynch(1:last_time_to_plot,:);
final_to_print = last_time_to_plot;
dataCase = 2; % train (0), validation (1) or test (2)
timeStepsBeforeEndForErrCalculation = 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Add output folder
outputFolder = paths.path_to_tracking_output;
addpath(outputFolder);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load GT data
load(strcat(outputFolder, '\GT_anomalies_and_curves_ES.mat'))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[data, testingLength, alphas_from_assignments_video, ...
    alphas_from_assignments_odometry, realParamsDenorm, ...
    videoPredOdometryDenorm, odometryUpdatedOdometryDenorm] = ...
    ExtractTrackingDebugInformation(paths,dataCase, endingPoint);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Find distances using particles weights
[all_distances, ~, mean_distances] = ...
    FindParticleDistancesFromCurrentRealValue(realParamsDenorm, videoPredOdometryDenorm);
% Use all_distances_od to plot only with nearest particles
[all_distances_od, ~, mean_distances_od] = ...
    FindParticleDistancesFromCurrentRealValue(realParamsDenorm, odometryUpdatedOdometryDenorm);
[summedDistancesOverTimeInstant,mean_distances_reweighted] = WeightParticlesDistancesAndFindMean(all_distances, data.particlesWeights);
[summedDistancesOverTimeInstant_od,mean_distances_reweighted_od] = WeightParticlesDistancesAndFindMean(all_distances_od, data.particlesWeights);
mean_distances_reweighted
median(summedDistancesOverTimeInstant)
mean_distances_reweighted_od
median(summedDistancesOverTimeInstant_od)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Find the distances with corrections done by the full resampling (video)
realParams = realParamsDenorm;
predParams = videoPredOdometryDenorm;
newIndicesForSwapping = data.newIndicesForSwapping;
whenRestarted = data.whenRestarted;
indicesRestartedParticles = data.indicesRestartedParticles;

[predParamsCorrectedUpdatedOdometry_v] = CutPredictionsBasedOnResamplingFullAndRestart(...
    predParams, newIndicesForSwapping, whenRestarted, indicesRestartedParticles);
[~, min_distances_v, mean_distances_v] = ...
    FindParticleDistancesFromCurrentRealValue(realParams, predParamsCorrectedUpdatedOdometry_v);
mean(mean_distances_v(1:end-timeStepsBeforeEndForErrCalculation))
median(mean_distances_v(1:end-timeStepsBeforeEndForErrCalculation))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Find the distances with corrections done by the full resampling (odometry)
realParams = realParamsDenorm;
predParams = odometryUpdatedOdometryDenorm;
newIndicesForSwapping = data.newIndicesForSwapping;
whenRestarted = data.whenRestarted;
indicesRestartedParticles = data.indicesRestartedParticles;

[predParamsCorrectedUpdatedOdometry_o] = CutPredictionsBasedOnResamplingFullAndRestart(...
    predParams, newIndicesForSwapping, whenRestarted, indicesRestartedParticles);
[all_distances, min_distances_od, mean_distances_od] = ...
    FindParticleDistancesFromCurrentRealValue(realParamsDenorm, predParamsCorrectedUpdatedOdometry_o);
mean(mean_distances_od(1:end-timeStepsBeforeEndForErrCalculation))
median(mean_distances_od(1:end-timeStepsBeforeEndForErrCalculation))

% Define nearest particles to the real value
[~, min_distances_indexes]=min(all_distances,[],2);
odometry_min_particles = zeros(size(predParamsCorrectedUpdatedOdometry_o, 1), ...
    size(predParamsCorrectedUpdatedOdometry_o, 3));
for i = 1:size(predParamsCorrectedUpdatedOdometry_o, 1)
    particle_index = min_distances_indexes(i);
    odometry_min_particles(i, 1) = predParamsCorrectedUpdatedOdometry_o(i, particle_index, 1);
    odometry_min_particles(i, 2) = predParamsCorrectedUpdatedOdometry_o(i, particle_index, 2);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOTTING
% tracking
initial_velocity = predParamsCorrectedUpdatedOdometry_o(T,:,:) - predParamsCorrectedUpdatedOdometry_o(T-1,:,:);
IMU_odometry = [];
for i = T:last_time_to_plot
    if i == T
        current_vel = initial_velocity;
    else
        current_vel = predParamsCorrectedUpdatedOdometry_o(i,:,:) - predParamsCorrectedUpdatedOdometry_o(i-1,:,:);
    end
    new_odometry = CalculateOdometryFromIMU(orientationSynch(i,:), linearAcceleration(i,:), ...
        predParamsCorrectedUpdatedOdometry_o(i,:,:), orientationSynch(T,:), initial_velocity, current_vel);
    IMU_odometry = [IMU_odometry, new_odometry(:,1)];

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
minX = min(realParams(:,1));
maxX = max(realParams(:,1));
minY = min(realParams(:,2));
maxY = max(realParams(:,2));
offset = 1;

tracking = figure
scatter3(realParams(1:last_time_to_plot,1), ...
         realParams(1:last_time_to_plot,2), ...
         1:1:last_time_to_plot,5,'b','filled')
hold on
for i = 1:size(predParamsCorrectedUpdatedOdometry_v, 2)
    
    s = scatter3(predParamsCorrectedUpdatedOdometry_v(1:last_time_to_plot,i,1), ...
                 predParamsCorrectedUpdatedOdometry_v(1:last_time_to_plot,i,2), ...
                 1:1:last_time_to_plot,5,'g','filled');
    if i > 1
        s.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
end
for i = 1:size(predParamsCorrectedUpdatedOdometry_o, 2)
    s = scatter3(predParamsCorrectedUpdatedOdometry_o(1:last_time_to_plot,i,1), ...
                 predParamsCorrectedUpdatedOdometry_o(1:last_time_to_plot,i,2), ...
                 1:1:last_time_to_plot,5,'r','filled');
    if i > 1
        s.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
end


xlabel('x (m)', 'FontSize',11, 'Interpreter','latex')
ylabel('y (m)', 'FontSize',11, 'Interpreter','latex')
zlabel('time', 'FontSize',11, 'Interpreter','latex')
legend({'Real positions', 'Estimated positions from direct video DBN', ...
    'Estimated positions from combined DBN'},  ...
    'FontSize',9, 'Interpreter','latex', ...
    'Location', 'southwest', ...
    'Orientation', 'vertical')
axis([minX - offset, maxX + offset, ...
      minY - offset, maxY + offset])

figure
s = scatter(odometry_min_particles(1:last_time_to_plot,1), ...
    odometry_min_particles(1:last_time_to_plot,2),'r');
hold on
scatter(realParams(1:last_time_to_plot,1), ...
         realParams(1:last_time_to_plot,2),'b')

figure
s = scatter3(odometry_min_particles(1:last_time_to_plot,1), ...
    odometry_min_particles(1:last_time_to_plot,2), 1:last_time_to_plot, ...
    'r', 'filled');
hold on
scatter3(realParams(1:last_time_to_plot,1), ...
    realParams(1:last_time_to_plot,2), 1:last_time_to_plot, ...
    'b', 'filled');
%ax = gca;
% Requires R2020a or later
%exportgraphics(ax,'Plot_tracking.png','Resolution',800) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOTTING
% tracking

minX = min(realParams(:,1));
maxX = max(realParams(:,1));
minY = min(realParams(:,2));
maxY = max(realParams(:,2));
offset = 1;

tracking = figure

plot3(realParams(1:last_time_to_plot,1), ...
         realParams(1:last_time_to_plot,2), ...
         1:1:last_time_to_plot,'b')
hold on

mean_predParamsCorrectedUpdatedOdometry_v = squeeze(mean(predParamsCorrectedUpdatedOdometry_v, 2));
plot3(mean_predParamsCorrectedUpdatedOdometry_v(1:last_time_to_plot,1), ...
      mean_predParamsCorrectedUpdatedOdometry_v(1:last_time_to_plot,2), ...
      1:1:last_time_to_plot,'g');
mean_predParamsCorrectedUpdatedOdometry_o = squeeze(mean(predParamsCorrectedUpdatedOdometry_o, 2));
plot3(mean_predParamsCorrectedUpdatedOdometry_o(1:last_time_to_plot,1), ...
      mean_predParamsCorrectedUpdatedOdometry_o(1:last_time_to_plot,2), ...
      1:1:last_time_to_plot,'r');
  
xlabel('x (m)', 'FontSize',11, 'Interpreter','latex')
ylabel('y (m)', 'FontSize',11, 'Interpreter','latex')
zlabel('time', 'FontSize',11, 'Interpreter','latex')
legend({'Real positions ($x_t^o$)', 'Estimated pos. from direct video DBN ($\hat{x}_t^o$)', ...
    'Estimated pos. from CDBN ($x_{t/t}^o$)'},  ...
    'FontSize',9, 'Interpreter','latex', ...
    'Location', 'north', ...
    'Orientation', 'vertical')
axis([minX - offset, maxX + offset, ...
      minY - offset, maxY + offset])
grid on

ax = gca;
% Requires R2020a or later
exportgraphics(ax,fullfile(paths.baseFolderPath, 'Plot_tracking_result.png'),'Resolution',800) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% errors

mean_value = mean([mean_distances_v; mean_distances_od]);
std_value  = std([mean_distances_v; mean_distances_od]);

errorsFigure = figure
errorsFigure.Position = [0,0,1350, 300]
area((GT == 1) * (mean_value+4*std_value),  'FaceColor',[1 0.8 0.8],'EdgeColor',[1 0.8 0.8]);
hold on
area((GT == 2) * (mean_value+4*std_value)  , 'FaceColor',[0.6 1 0.6],'EdgeColor',[0.6 1 0.6]);
area((GT == 3) * (mean_value+4*std_value)   , 'FaceColor',[0.6 1 1],'EdgeColor',[0.5 1 1]);
plot(mean_distances_v(1:last_time_to_plot), 'g')
plot(mean_distances_od(1:last_time_to_plot), 'r')
grid on 
axis ([0 last_time_to_plot 0 (mean_value+4*std_value)])

xlabel('time', 'FontSize',15, 'Interpreter','latex')
ylabel('Errors (m)', 'FontSize',15, 'Interpreter','latex')

legend({'Pedestrian visible', 'Car restart','Oscillating motion', ...
    'Errors from direct video DBN', ...
    'Errors from combined DBN'},  ...
    'FontSize',15, 'Interpreter','latex', ...
    'Location', 'north', ...
    'Orientation', 'horizontal')

ax = gca;
% Requires R2020a or later
exportgraphics(ax,'Plot_errors.png','Resolution',800) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% anomalies

% take out all anomalies
abn_KLDA = data.anomalies(:,1);
abn_Video = data.anomalies(:,2);
%abn_Odom = data.anomalies(:,1);
abn_Rec = data.anomalies(:,3);
clusterLikelihoods = data.anomalies(:,4);
diff_in_predictions = data.anomalies(:,5);

% Find mean and min for anomalies that are across particles
mean_abn_KLDA = mean(abn_KLDA, 2);
min_abn_KLDA = min(abn_KLDA, [], 2);
max_abn_KLDA = max(abn_KLDA, [], 2);

mean_abn_Video = mean(abn_Video, 2);
min_abn_Video = min(abn_Video, [], 2);
max_abn_Video = max(abn_Video, [], 2);

%{
mean_abn_Odom = mean(abn_Odom, 2);
min_abn_Odom = min(abn_Odom, [], 2);
max_abn_Odom = max(abn_Odom, [], 2);
%}

mean_diff_in_predictions = mean(squeeze(mean(diff_in_predictions, 2)),2);
min_diff_in_predictions = min(squeeze(min(diff_in_predictions, [], 2)),2);

mean_clusterLikelihoods = mean(clusterLikelihoods, 2);
min_clusterLikelihoods = min(clusterLikelihoods, [], 2);
entropy_clusterLikelihoods = zeros(size(clusterLikelihoods, 1),1);
for i = 1:size(clusterLikelihoods, 1)
 entropy_clusterLikelihoods(i) = entropy(double(squeeze(clusterLikelihoods(i,:))));
end


begin = 40;
ending = 6000;

% Normalize anomalies

abn_KLDA_norm = (abn_KLDA - min(abn_KLDA(begin:(final_to_print-1))))/(max(abn_KLDA(begin:(final_to_print-1)))- min(abn_KLDA(begin:(final_to_print-1))));


%mean_abn_KLDAs_norm = (mean_abn_KLDAs - min(mean_abn_KLDAs(begin:final_to_print)))/(max(mean_abn_KLDAs(begin:final_to_print))- min(mean_abn_KLDAs(begin:final_to_print)));
%min_abn_KLDAs_norm = (min_abn_KLDAs - min(min_abn_KLDAs(begin:final_to_print)))/(max(min_abn_KLDAs(begin:final_to_print))- min(min_abn_KLDAs(begin:final_to_print)));

mean_abn_Video_norm = (mean_abn_Video - min(mean_abn_Video(begin:(final_to_print-1))))/(max(mean_abn_Video(begin:(final_to_print-1)))- min(mean_abn_Video(begin:(final_to_print-1))));
min_abn_Video_norm = (min_abn_Video - min(min_abn_Video(begin:(final_to_print-1))))/(max(min_abn_Video(begin:(final_to_print-1)))- min(min_abn_Video(begin:(final_to_print-1))));

%{
mean_abn_Odom_norm = (mean_abn_Odom - min(mean_abn_Odom(begin:final_to_print)))/(max(mean_abn_Odom(begin:final_to_print))- min(mean_abn_Odom(begin:final_to_print)));
min_abn_Odom_norm = (min_abn_Odom - min(min_abn_Odom(begin:final_to_print)))/(max(min_abn_Odom(begin:final_to_print))- min(min_abn_Odom(begin:final_to_print)));
%}

abn_Rec_norm = (abn_Rec - min(abn_Rec))/(max(abn_Rec)- min(abn_Rec));

mean_diff_in_predictions_norm = (mean_diff_in_predictions - min(mean_diff_in_predictions(begin:(final_to_print-1))))/(max(mean_diff_in_predictions(begin:(final_to_print-1)))- min(mean_diff_in_predictions(begin:(final_to_print-1))));
min_diff_in_predictions_norm = (min_diff_in_predictions - min(min_diff_in_predictions(begin:(final_to_print-1))))/(max(min_diff_in_predictions(begin:(final_to_print-1)))- min(min_diff_in_predictions(begin:(final_to_print-1))));

mean_clusterLikelihoods_norm = (mean_clusterLikelihoods - min(mean_clusterLikelihoods(begin:(final_to_print-1))))/(max(mean_clusterLikelihoods(begin:(final_to_print-1)))- min(mean_clusterLikelihoods(begin:(final_to_print-1))));
min_clusterLikelihoods_norm = (min_clusterLikelihoods - min(min_clusterLikelihoods(begin:(final_to_print-1))))/(max(min_clusterLikelihoods(begin:(final_to_print-1)))- min(min_clusterLikelihoods(begin:(final_to_print-1))));
entropy_clusterLikelihoods_norm = (entropy_clusterLikelihoods - min(entropy_clusterLikelihoods(begin:(final_to_print-1))))/(max(entropy_clusterLikelihoods(begin:(final_to_print-1)))- min(entropy_clusterLikelihoods(begin:(final_to_print-1))));


% Plotting

whenRestarted = data.whenRestarted;

anomaliesFigure = figure
anomaliesFigure.Position = [0,0,1350, 300]
hold on
plot(smooth(abn_KLDA_norm(1:(last_time_to_plot-1))), 'm')
%plot(min_abn_KLDAs_norm(1:last_time_to_plot), 'm')
%plot(mean_abn_Video_norm(1:last_time_to_plot), 'g')
%plot(smooth(min_abn_Video_norm(1:last_time_to_plot)), 'g')
%plot(min_abn_Odom_norm(1:last_time_to_plot), 'g')
plot(smooth(abn_Rec_norm(1:(last_time_to_plot-1))), 'r')
%plot(smooth(mean_diff_in_predictions_norm(1:last_time_to_plot)), 'b')
plot(smooth(min_clusterLikelihoods_norm), 'c')
%scatter(whenRestarted, ones(length(whenRestarted),1), 'k')
grid on 
axis ([0 (last_time_to_plot-1) 0 1])

xlabel('time', 'FontSize',15, 'Interpreter','latex')
ylabel('Abnormalities', 'FontSize',15, 'Interpreter','latex')
legend({'$\tilde{S}_t$ abn.', 'rec. abn', ...
    'min $\tilde{S}_t$ likelihood'}, ...
    'FontSize',15, 'Interpreter','latex', 'Location', 'north', ...
    'Orientation', 'horizontal')



%{
    anomaliesFigure = figure
anomaliesFigure.Position = [0,0,1350, 300]
hold on
plot(smooth(abn_KLDA_norm(1:last_time_to_plot)), 'm')
%plot(min_abn_KLDAs_norm(1:last_time_to_plot), 'm')
plot(mean_abn_Video_norm(1:last_time_to_plot), 'g')
%plot(smooth(min_abn_Video_norm(1:last_time_to_plot)), 'g')
%plot(min_abn_Odom_norm(1:last_time_to_plot), 'g')
plot(smooth(abn_Rec_norm(1:last_time_to_plot)), 'r')
plot(smooth(mean_diff_in_predictions_norm(1:last_time_to_plot)), 'b')
plot(smooth(min_clusterLikelihoods_norm), 'c')
scatter(whenRestarted, ones(length(whenRestarted),1), 'k')
grid on 
axis ([0 last_time_to_plot 0 1])

xlabel('time', 'FontSize',15, 'Interpreter','latex')
ylabel('Abnormalities', 'FontSize',15, 'Interpreter','latex')
legend({'$\tilde{S}_t$ abn.', '$a_t abn.$', 'rec. abn', ...
    'pred. difference abn.', 'min $\tilde{S}_t$ likelihood'}, ...
    'FontSize',15, 'Interpreter','latex', 'Location', 'north', ...
    'Orientation', 'horizontal')
%}
    
    
    
% load ('C:\Users\asus\Documents\Datasets_and_results\Icab_SM_3\GT_anomalies_and_curves.mat')
% 
% GTatRestartingPoints = GT(whenRestarted);
% anomaliesAtRestartingPoints = sum(GTatRestartingPoints==1) + sum(GTatRestartingPoints==2) + sum(GTatRestartingPoints==3);
% percentage_anomaliesAtRestartingPoints = anomaliesAtRestartingPoints/length(whenRestarted)*100
% 
% abnormalities_count = sum(GT==1) + sum(GT==2) + sum(GT==3);
% total_count =  sum(GT==0) + sum(GT==1) + sum(GT==2) + sum(GT==3) + sum(GT==4) + sum(GT==5);
% percentage_abnormalities = abnormalities_count/total_count*100


%%
% FigureGTOdometry = figure;
% FigureGTOdometry.Position = [0,0,1500, 300];

% GT
% Anomaly defined by me
% area([(GT == 1)]    , 'FaceColor',[1 0.8 0.8],'EdgeColor',[1 0.8 0.8]);
% hold on
% area([(GT == 2)]    , 'FaceColor',[0.6 1 0.6],'EdgeColor',[0.6 1 0.6]);
% area([(GT == 3)]    , 'FaceColor',[0.6 1 1],'EdgeColor',[0.5 1 1]);
% %area([(GT == 5)]    , 'FaceColor',[1 0.7 1],'EdgeColor',[1 0.7 1]);
% scatter(whenRestarted, ones(length(whenRestarted),1)*0.35, 50, 'k', 'filled')
% xline(whenRestarted,'LineWidth',1,'LineStyle', ':','color','k')
% axis([0 size(GT, 2) 0 1])
% %set(gca,'xtick',[])
% %xlabel('time instant $t$','FontSize',11, 'Interpreter','latex')
% xlabel('time','FontSize',15, 'Interpreter','latex')
% %ylabel('$|z_{t+1|t}^v - z_{t+1}^v|$','FontSize',12, 'Interpreter','latex')
% %legend('$err_{a^v_t}$', '$err_{z^v_t}$','FontSize',9, 'Interpreter','latex')
% legend({'Pedestrian visible', 'Car restart', 'Oscillating motion', ...
%         'Particles restarts'},  ...
%         'FontSize',15, 'Interpreter','latex', 'Location', 'south', ...
%         'Orientation', 'horizontal')
%     
% ax = gca;
% resolution = 1200;
% % Requires R2020a or later
% exportgraphics(ax, ...
%     'C:\Users\asus\Documents\Datasets_and_results\Icab_SM_3\StartingPointsVsAnomalies.png', ...
%     'Resolution',resolution) 
