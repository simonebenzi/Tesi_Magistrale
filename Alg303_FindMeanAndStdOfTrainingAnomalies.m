
function [] = Alg303_FindMeanAndStdOfTrainingAnomalies()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters for path
baseFolderPath = LoadBaseDataFolder();
paths          = DefinePathsToData(baseFolderPath);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Anomalies
[anomalies, ~] = loadObjectGivenFileName(paths.path_to_training_anomalies);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Finding the means and stds of the anomalies
anomalyMeans = [];
anomalyStds  = [];
numberOfAnomalies = size(anomalies,2);
stdTimes = 3;
repetitions = 200;
for indexOnAbn = 1:numberOfAnomalies
    anomaly = anomalies(:,indexOnAbn);
    % Eliminate outliers
    anomaly = CutArraOutOfMeanStdThresh(anomaly, stdTimes, repetitions);
    % Find mean and std
    anomalyMean = mean(anomaly);
    anomalyStd = std(anomaly);
    % Insert the mean and std into the vector
    anomalyMeans = [anomalyMeans, anomalyMean];
    anomalyStds  = [anomalyStds, anomalyStd];
    % 2.5 standard deviations away from norm, for plotting
    threshExample = anomalyMean + 2.5*anomalyStd;
    figure
    hold on
    plot(anomaly)
    line([0 length(anomaly)], [threshExample threshExample]);
    title('Training anomaly')
    xlabel('time')
    ylabel('anomaly')
end
%% Making the lists of means and stds into a string
anomalyMeansString = '';
anomalyStdsString  = '';
for indexOnAbn = 1:numberOfAnomalies
    anomalyMean = anomalyMeans(indexOnAbn);
    anomalyStd  = anomalyStds(indexOnAbn);
    anomalyMeansString = [anomalyMeansString, num2str(anomalyMean)];
    anomalyStdsString  = [anomalyStdsString, num2str(anomalyStd)];
    if indexOnAbn ~= numberOfAnomalies
        anomalyMeansString = [anomalyMeansString, ','];
        anomalyStdsString = [anomalyStdsString, ','];
    end
end
%% Saving means and stds in the configuration file
parametersNames = char('AnomaliesMeans', 'AnomaliesStandardDeviations');
parametersValues = char(anomalyMeansString, anomalyStdsString);
SaveConfigurationValuesToJSONFile(paths.path_to_config_KVAE_combined, ...
    parametersNames, parametersValues)
end
