function [files] = ...
    DefineAlphasMusPredictedMusFilesDependingOnDataAndModelCase(...
    model,dataCase, files)

% Estimations from training code
if model == 1
    files.alphaFile               = 'alphas50.mat';
    files.predictedParamsFile     = 'predicted_params50.mat';
    files.musFile                 = 'FILTER_filter51.mat';

elseif model == 2
    if dataCase == 0
        files.alphaFile               = 'alphas60.mat';
        files.predictedParamsFile     = 'predicted_params60.mat';
        files.musFile                 = 'filter60.mat';
    elseif dataCase == 1
        files.alphaFile               = 'alphas60.mat';
        files.predictedParamsFile     = 'predicted_params60.mat';
        files.musFile                 = 'filter60.mat';
    end
elseif model ==3
    if dataCase == 0
        files.alphaFile               = 'alphas59.mat';
        files.predictedParamsFile     = 'predicted_params59.mat';
        files.musFile                 = 'smooth59.mat';
    elseif dataCase == 1
        files.alphaFile               = 'alphas59.mat';
        files.predictedParamsFile     = 'predicted_params59.mat';
        files.musFile                 = 'smooth59.mat';
    end
elseif model == 4
    if dataCase == 2
        files.alphaFile               = 'alphas39.mat';
        files.predictedParamsFile     = 'predicted_params39.mat';
        files.musFile                 = 'smooth39.mat';
    elseif dataCase == 3
        files.alphaFile               = 'alphas39.mat';
        files.predictedParamsFile     = 'predicted_params39.mat';
        files.musFile                 = 'smooth39.mat';
    end
else
    files.alphaFile               = 'TRAIN_single_epoch_alphas.mat';
    files.predictedParamsFile     = 'TRAIN_single_epoch_predicted_params_min.mat';
    files.musFile                 = 'TRAIN_single_epoch_a_states.mat';
end

end