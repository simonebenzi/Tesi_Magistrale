
% This function allows to find the covariance of the odometry prediction and the
% covariance of the prediction of the D matrices.
% INPUTS:
% - videoVocabulary: the video vocabulary calculated thus far;
% - odometryVocabulary: the odometry vocabulary. Be careful that in the 
%   processing of the clusters, some clusters might have been eliminated, 
%   so check to have performed 'ModifyOdometryClusteringBasedOnCutClusters'
%   before, if that was the case.
% - trainingGSs: features/odometry GSs.
% - predictedParamsKVAE: predicted features/odometry GSs from KVAE.
% OUTPUTS:
% - net: the video vocabulary now inclusive of the following two additional
%   fields:
%   > nodesCovPred: covariance over odometry clusters prediction;
%   > nodesCovD: covariance over the positioning using D matrices;
%   net is also saved as a .mat file in 'folder' with the name of 
%   'Odometry_based_vocabulary.mat'.
function[videoVocabulary] = FindCovariancePredictionVsDMatrices(videoVocabulary, ...
    odometryVocabulary, trainingGSs, predictedParamsKVAE)

    %% Finding the prediction errors using the mean of the clusters    
    prediction_errors             = zeros(size(trainingGSs,1), size(trainingGSs,2));
    prediction_errors_per_cluster = cell(videoVocabulary.N,1);    
    for i = 1: size(trainingGSs, 1) - 1
        if not(ismember(i+1,videoVocabulary.startingPoints) )      
            % Odometry value
            current_odometry_point  = trainingGSs(i,:);
            next_odometry_point     = trainingGSs(i+1,:);           
            % Cluster at current time instant
            current_cluster         = odometryVocabulary.dataColorNode(i);
            % Node mean of current odometry cluster
            current_nodes_mean      = odometryVocabulary.nodesMean(current_cluster,:);            
            % Prediction
            current_prediction(1:2) = current_odometry_point(1:2) + current_nodes_mean(3:4);
            current_prediction(3:4) = current_nodes_mean(3:4);            
            % Prediction error
            current_prediction_err  = next_odometry_point - current_prediction;           
            prediction_errors(i,:)  = current_prediction_err;
            prediction_errors_per_cluster{current_cluster,1} = ...
                                     [prediction_errors_per_cluster{current_cluster,1}; ...
                                     current_prediction_err];
        end
    end 
    %% Finding the D matrix prediction errors   
    prediction_D_errors             = zeros(size(predictedParamsKVAE,1), size(trainingGSs,2));
    prediction_D_errors_per_cluster = cell(videoVocabulary.N,1);    
    for i = 1: size(predictedParamsKVAE, 1) - 1       
        % Odometry value
        current_odometry_point  = trainingGSs(i,:);    
        % Cluster at current time instant
        current_cluster         = odometryVocabulary.dataColorNode(i);        
        % Prediction
        current_prediction      = predictedParamsKVAE(i,:);       
        % Prediction error
        current_prediction_err  = current_odometry_point - current_prediction;   
        prediction_D_errors(i,:)= current_prediction_err;       
        prediction_D_errors_per_cluster{current_cluster,1} = ...
                                 [prediction_D_errors_per_cluster{current_cluster,1}; ...
                                 current_prediction_err];
    end
    %% Find covariance matrix for prediction from cluster means    
    for i = 1:videoVocabulary.N
        nodesCovPred{1,i}  = cov(prediction_errors_per_cluster{i,1}, 1);
        nodesCovD{1,i}     = cov(prediction_D_errors_per_cluster{i,1}, 1);
    end    
    videoVocabulary.nodesCovPred = nodesCovPred;
    videoVocabulary.nodesCovD    = nodesCovD;
end


