
Configuration values for odometry-from-video testing:

- "config_Filename": name of the file where to save the JSON/txt configuration file. Do not put the extension here;
- "clustering_data_file_odom" : path to .mat file containg the odometry clustering;
- "clustering_data_file_video": path to .mat file containg the video clustering;
- "kvae_file": path to .torch file of the trained CG-KVAE_D;
- "N_Particles": how many particles you want the combined CG-KVAE / MJPF to have;
- "skew_video": temperature value of video. Used only in the first time instant;
- "output_folder": path where you want to save your results;
- "type_of_weighting": for now only 0 (video prediction anomaly + clusters likelihood) is used.
- "known_starting_point": do we know the positional point where we are starting from?
- "resampleThresh": threshold on Neff (number of effective samples) for performing resampling,
- "firstResampleThresh": threshold on Neff when the code starts, until the first resampling itself.
  After the first resampling, it switches to 'resamplingThreshold';
- "observationVariance": observation variance;
- "testingOnTraining": 'true' if you track the training data, 'false' if you track the testing data. Leave it to False!
- "initialTimeInstant": time instant of data where to begin the tracking;
- "lastTimeInstant": time instant of data where to end the tracking. If you want to take the last, put -1;
- "transitionMatExplorationPercentage": how much we use the 'transitionMatExploration' when performing next particle prediction. 
     If this is seto to 0, the matrix is not used. Leave it to 0 for now! This is for future development.
- "saveReconstructedImages": do you want to save the images reconstructed from the VAE?
- "reconstructedImagesFolder": where to save the reconstructed images.
- "fastDistanceCalculation": distance of latent states 'a' from clusters calculated with MSE instead of Bhattacharya?
- "usingAnomalyThresholds": are we using particles restarting?
- "percentageParticlesToReinitialize": what is the percentage of "numberOfParticles" to restart if 
  "usingAnomalyThresholds" is set to True?
- "AnomaliesMeans", "AnomaliesStandardDeviations": mean and std of training anomalies.
- "stdTimes": if "usingAnomalyThresholds = True", how to set the anomaly threshold w.r.t. anomaly means and stds,
   i.e. threholds = AnomaliesMeans + stdTimes*AnomaliesStandardDeviations?
- "time_window": time window of anomaly calculation.
- "time_enough_ratio": if time_enough_ratio*time_window inside time_window is over the anomaly threshold, the 
   data is considered as abnormal.
- "time_wait_ratio": some anomalies should signal a restart only when they are back to a normal value. For example,
   if we restart particles when a high reconstruction anomaly is present, we will probably end up in a wrong zone.
   So, after at least time_enough_ratio*time_window abnormal points have been detected in the time window, before 
   restarting the particles we wait that abnormal points are less than time_wait_ratio*time_window in the time_window.
- "obs": how much we favor D/E matrix prediction (the observation) w.r.t. the prediction of the odometry dynamics 
   model. 
