# SIMULTANEOUS LOCALIZATION AND ANOMALY DETECTION FROM FIRST-PERSON VIDEO DATA THROUGH A COUPLED DYNAMIC BAYESIAN NETWORK MODEL

This folder contains the codes for performing training and testing for:

- CG-KVAE + MJPF for visual based localization / visual odometry and anomaly detection.
  The code is written in pytorch and in MATLAB.

Part of the code considers the extraction of data from the Carla simulator, but it can be skipped depending on which dataset is used.

## Python version and Required installations

Use **python 3.7** for use with **Carla 0.9.10**. To create an environment for Anaconda with python 3.7:

```
conda create -n name -c conda-forge spyder python=3.7
```

where 'name' is the name to give to the environment.

The **required installations** for **python** are:

pytorch and torchvision:

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

torchsummary

```
pip install torchsummary
```

scipy

```
conda install scipy
```

mat4py

```
pip install mat4py
```

matplotlib

```
conda install matplotlib
```

numpy

random

json

os

spyder-kernels

```
conda install spyder-kernels
```

```
pip install spyder-kernels==2.2
```

In the case in which the **Carla dataset** is also used:

```
pip install pygame
conda install networkx
```

The **add-on tollboxes** required for **MATLAB** are:

- DSP System Toolbox

Read here how to install MATLAB adds-on:

https://it.mathworks.com/help/matlab/matlab_env/get-add-ons.html

## Dataset

You have to insert training, validation (either together or separated) and testing data.

You can choose whether extracting the dataset from the Carla simulator or using a dataset of your choice.

**For the TRAINING/VALIDATION data:**

If you are using Carla, the simulator files need to be downloaded in a separate folder. You must run the `Alg000_ov_ExtractCarlaTrajectoriesAutomatically.py` and `Alg001_o_JoinPositionsFromCarlaTrajectories.py`. (see <u>Optional Carla Codes</u> section)

If you are using another dataset, you must create a separate folder where to insert it.

<u>If you have a single dataset with multiple training+validation trajectories, and you want for the code to divide it in two parts</u>, given a ratio you give, create two folders called:

- 'camera', where each data trajectory must be inserted in a separate subfolder. Be careful that individual images in the subfolders are named in such a way that they can be loaded with the correct order (i.e., DON'T name them '1.png', '2.png', ... '100.png', but name them instead '0001.png', '0002.png', ..., '0100.png'). Images must be **RGB**, either '.png' or '.jpeg'.
- 'position', which must include a matlab object 'train_positions.mat'. If there are N trajectories, this object must be structured as a *cell array of dimension (1, N). Inside each cell, trajectories must be structured as (M,2), where M is the length of the trajectory.* Position along x and position along y must be provided at each time instant, as can be seen from the value 2 in (M,2).

To summarize, the dataset folder must be structured as:

```
baseFolder\
   image\
      traj_0001\
         img_0001.png
         img_0002.png
         ...
         img_0100.png
      traj_0002\
         img_0001.png
         img_0002.png
         ...
         img_0100.png
   position\
      train_positions.mat
```

To note that the subfolders of the 'image' folder (i.e., 'traj_0001' and 'traj_0002' above) and the images inside do not need to have any specific name.

In the Carla case, the code will automatically produce a folder as the one above.

<u>If you already have a dataset divided in training and validation,</u> instead of the 'image' and 'position' folders, create 'train_images', 'train_positions' and 'validation_images', 'validation_positions' folders. Define:

```
separateTrainingAndValidation = false;
```

in `Alg___PerformTraining.m`, i.e., `Alg100_ov_Divide_in_train_and_validation.m` will be skipped.

<u>If your training+validation dataset is composed by a single long trajectory</u>, you have to split it manually, as the code for separating between training and validation divides based on number of trajectories and not based on data length. Again set `separateTrainingAndValidation = false`.

If your validation dataset is composed by very few trajectories (or even just one trajectory), the testing over the multiple parameters might not provide a good result for what concerns the `firstResampleThresh `parameter, in the case testing were to start from a different initial position w.r.t. training and validation. In this case, it is suggested to set in `Config_KVAE_combined`:

```
 "addingInitialTimeInstantsOnVal" : true,
```

This allows to test on validation starting from different points. You must also set `newInitialTimeInstantsFrequency `to define how many new starting points w.r.t. the number of datapoints you wish.

**For the TESTING data:**

The folders 'image' and 'position' are the ones related to the training data. You have also to create two similar folders for testing, called 'test_images' and 'test_position' (with test_positions.mat).

If you are using Carla and running either `Alg002_ov_ExtractFreeManualTrajectoriesCarla.py` or `Alg003_ov_ExtractManualCarlaTrajectoryFromStartToDest.py`, these codes will automatically create the 'test_images' and 'test_position' folders, with the correct structure. Additionally, they will create a 'On_the_fly_test_images' and a 'On_the_fly_test_positions' folders, that can be used for testing just the last extracted trajectory (i.e., to use for demos).

## Optional Carla codes

!! The Carla version to be used is **Carla 0.9.10** !!

Some Carla codes are given, following a similar structure to the examples 'manual_control' and 'automatic_control' provided by Carla.

The following codes related to Carla can be run to build Carla trajectories for training and testing:

**For automatic TRAINING DATA extraction:**

- (OPTIONAL) `Alg000_ov_ExtractCarlaTrajectoriesAutomatically.py` is for extracting the dataset from the Carla simulator in an automatic way. Several global parameters can be defined in the higher part of the code. Before running this code, define the path to the Carla folder in the file 'CarlaFolder' under the 'ConfigurationFiles' directory.

- (OPTIONAL) `Alg001_o_JoinPositionsFromCarlaTrajectories.m` MUST be performed after `Alg000_ov_ExtractCarlaTrajectoriesAutomatically.py`, if this is used. It restructures the positional data obtained from the above code, so that it is saved as a single cell array.

**For manual TESTING DATA extraction:**

- (OPTIONAL) `Alg002_ov_ExtractFreeManualTrajectoriesCarla.py` is a code for extracting trajectories from Carla, while getting acquainted with the simulator. It allows to drive around one of the Carla Towns, starting from random spawn points. 
  Several global parameters can be defined in the higher part of the code.
  Before running this code, define the path to the Carla folder in the file 'CarlaFolder'
  under the 'ConfigurationFiles' directory. This was the code used for letting people get acquainted with Carla at the GTTI meeting.

- (OPTIONAL) `Alg003_ov_ExtractManualCarlaTrajectoryFromStartToDest.py` is for extracting a manual trajectory from the Carla simulator, starting from a fixed spawn point, until a destination is reached. Several global parameters can be defined in the higher part of the code, including the start and destination points. Before running this code, define the path to the Carla folder in the file 'CarlaFolder' under the 'ConfigurationFiles' directory. This was the code used for final testing data extraction at the GTTI meeting.

The first two of these codes is used for extracting the TRAINING data, whereas the last two can be used for extracting the TESTING data.

The outputs from the `Alg000_ov_ExtractCarlaTrajectoriesAutomatically.py` will be placed in the 'position' and 'camera' folders. The 'camera' folder is structured as desired. However, the 'position' will have each trajectory saved as a different .mat file. They must be joined in a single cell array, and this is what `Alg001_o_JoinPositionsFromCarlaTrajectories.m` does.

The outputs from `Alg002_ov_ExtractFreeManualTrajectoriesCarla.py` and `Alg003_ov_ExtractManualCarlaTrajectoryFromStartToDest.py` will be placed in the dataset directory under 'test_images', 'test_position', 'On_the_fly_test_images' and 'On_the_fly_test_positions'. The former two folders include the data for all runs; the latter two for the last run only.

## How to run the code once the dataset is ready

To run the entire code (except the Carla part), run the MATLAB file `Alg___PerformAllSteps.m`.

First, however, there are several parameters you need to define.

## Parameters and paths to define

Parameters to be defined are placed in the 'ConfigurationFiles' folder.

Inside this folder, you can also find some README files with the description of all the parameters.

Parameters and paths to be set for using this code:

- (OPTIONAL) In the file 'CarlaFolder' insert the path to folder where the Carla codes are located. Before using the code, check that the Carla egg file is present and that it is the correct one (i.e., correct python and windows version). The egg file is typically extracted under '.../PythonAPI/carla/dist' in the Carla folder, for Windows.

- In the file 'BaseDataFolder' insert the path to the dataset directory. If you are choosing to extract data from Carla, this is where the extracted data will be placed. Otherwise, you must insert your dataset under this directory.

- In the code `Config_data_partitioning.m` define  what percentage of the trajectories under the 'image' and 'position' dataset folders will be used for training (the rest will be kept for validation). This is for the case in which your dataset is not already divided in a training and validation part.

- In the code `Config_filtering.m` define the parameters for performing the filtering of the positional state. These will be used in the code `Alg102_o_ExtractOdometryParametersForTrainValAndTest.m.`

- In the code `Config_clustering.m` define the parameters for performing the GNG clustering.  These will be used in the codes `Alg207_ov_PerformClustering.m` and `Alg210_Final_VideoClusteringExtraction.m`.

- In the file` Config_VAE.json` define the parameters of the VAE.  These will be used in the code `Alg205_v_TrainVAE.p`y.

- In the file `Config_KVAE.json` define the parameters of the KVAE. These will be used in the code `Alg209_ov_TrainKVAE.py`.

- In the file `Config_KVAE_combined.json` define the parameters of the combined MJPF for Visual-Based Localization. Some of these parameters will be set automatically by the algorithm and will be saved here.  One important parameter to be set is` usingAnomalyThresholds`. If you want to find the best parameters for particles restarting on validation, and then apply it on testing, set:
  
  `usingAnomalyThresholds = true`
  
  otherwise set it to false.
  
  Another important parameter is `known_starting_point,` which defines whether we suppose to know the position of where we begin the tracking or not. If we know this position, set:
  
  `known_starting_point = true`
  
  otherwise set it to false.

- In the file `Config_KVAE_combined_multiple_parameters.json` write the different values of the tracking parameters that you want the algorithm to try on the validation dataset. The ones providing the best results will be saved in `Config_KVAE_combined.json` and used on the testing dataset too.

- In the file `Config_KVAE_combined_multiple_thresholds.json` write the different values of the threshold-related parameters that you want the algorithm to try on the validation dataset. The ones providing the best results will be saved in `Config_KVAE_combined.json` and used on the testing dataset too. 

## Code names

Codes have names with a number of three digits, and the letters 'o', 'v' or 'ov'.

First digit from the left:
0: optional Carla codes for dataset extraction.
1: codes for dataset pre-elaboration and plotting.
2: training codes
3: testing codes

Letters:
'o' means that the code is handling only the odometry data.
'v' means that the code is handling only the video data.
'ov' means that the code is handling both the odometry and the video data.

The code `Alg___PerformAllSteps.m` does not have the above numbers or letters. It can be used to call ALL the pre-elaboration, training and testing codes.

## MATLAB paths

The 'MATLAB_paths' folder contains two important functions:

- 'AddAdditionalNecessaryPaths' allows to add ALL the necessary code paths, when using MATLAB. It is called in ALL the MATLAB codes in the base folder.

- 'DefinePathsToData' creates a structure which contains ALL the paths to each used MATLAB object in the dataset folder. It is called in ALL the MATLAB codes in the base folder.

## Pre-elaboration, training and testing

First, the dataset is pre-elaborated (codes 1).

Then, the pre-elaborated data are used for performing the training of the models (codes 2). 

Finally, testing (codes 3) is performed on validation data, on training data, on validation data again and finally on test data. 

In the first three testing steps parameters of testing are chosen.

To execute all the steps related to pre-elaboration you can call the `Alg___PerformAllSteps.m `function.

Otherwise, if for some reason you have to restart from a specific point, you can locate it inside `Alg___PerformAllSteps.m` and continue from then onwards. 

If just one of the steps must be performed again (e.g., if the testing data was not initially available), you can run the single necessary code from the base folder.

## Pre-elaboration of the data

Before starting the pre-elaboration, the dataset must be ready. This means that you have either extracted it with Carla, or inserted it in a folder as defined in the 'DATASET' 
section.

It is not necessary that you have the testing dataset ready. If it is not ready, the code will go on with the pre-elaboration (and, in the next section, with the training). 

Notice, however, that you will have to go back and manually re-run the codes `Alg101_o_NormalizeOdometryDataForTrainValAndTest.m`, `Alg102_o_ExtractOdometryParametersForTrainValAndTest.m` and `Alg208_ov_ExtractDataFileWithVideoAndParametersSingleTrajectory.py`, so that also the testing data will be correctly elaborated.

Pre-elaboration allows you to prepare your data before moving on to the actual training. This means separating training and validation, normalizing the positional data and filtering it. Additionally, some functions are available for plotting your dataset, so that you can examine it before moving to the training and testing parts.

Codes related to pre-elaboration are::

- `Alg100_ov_Divide_in_train_and_validation.m`: separates training and testing according to the ratio defined in `Config_data_partitioning.m`.
  *Inputs from: 'image', 'positions'.
  Outputs to: 'train_images', 'train_positions', 'validation_images', 'validation_positions'.*

- `Alg101_o_NormalizeOdometryDataForTrainValAndTest.m`: normalizes the training, validation and testing data.
  *Inputs from: 'train_positions', 'validation_positions', 'test_positions'.
  Outputs to: the same folders of input, generating 'train_positions_max.mat', 'train_positions_min.mat', 'train_positions_norm.mat'.*

- `Alg102_o_ExtractOdometryParametersForTrainValAndTest.m`: filters the training, validation and testing data and extracts the Generalized States.
  *Inputs from: 'train_positions', 'validation_positions', 'test_positions'.
  Outputs to: 'train_GSs', 'validation_GSs', 'test_GSs'.*

- `Alg103_o_PlotTrajectories.m`: the odometry trajectories are plotted in 2D and are saved as a .png file, at the base of the dataset folder.
  *Inputs from: 'train_GSs', 'validation_GSs', 'test_GSs'.
  Outputs to: 'Odometry_plot.png' in the base dataset folder.*

- `Alg104_ov_MakeVideoOfTrajectory.m`: odometry and image data are used to create a video, that is saved at the base of the dataset folder.
  *Inputs from: 'train_GSs', 'validation_GSs', 'test_GSs', 'train_images', 'validation_images', 'test_images'.
  Outputs to: '{}.mp4' in the base dataset folder, where {} is the name of the dataset folder.*

## Training of the models

The training of the model proceeds through the following steps:

- A VAE is trained on the image data;

- The clustering is extracted, produced using the odometry data;

- A KVAE is trained, starting from the learned VAE and using the obtained clustering;

- The parameters (means and covariances) of the video clusters are extracted.

The codes performing the above steps are the following:

- `Alg205_v_TrainVAE.py` : a VAE for reconstructing the training images is learned. 
  The configuration parameters of the VAE must be inserted in the file `Config_VAE.json` under the 'ConfigurationFiles' folder. The VAE is trained for as many epochs as defined in the configuration file. Then, the model of the epoch which granted the best loss on the validation dataset is selected. Finally, the latent states of the training data are extracted for this selected VAE and are saved.
  *Inputs from: 'train_images', 'validation_images'.
  Outputs to: 'trained_VAE'.*

- `Alg206_ov_AlignOutputOfVAEWithOdometry.m`: a code for aligning the latent states obtained from the VAE with the odometry GSs.
  *Inputs from: 'trained_VAE'.
  Outputs to: 'trained_VAE'.*

- `Alg207_ov_PerformClustering.m`: clustering is performed. The parameters for clustering must be defined in the `Config_clustering.m `code under the 'ConfigurationFiles' folder. There are several clustering choices that can be taken:
  
  - clustering on the odometry GSs;
  
  - clustering on the odometry GSs, giving more importance to the orthogonal component  w.r.t. direction of motion (this is the one chosen for the paper);
  
  - clustering combining the odometry GSs and the latent states from the VAE.
  
  For now, it is not possible to cluster using the latent states from the VAE together with the odometry GSs with higher importance to the orthogonal component. Optimization on the clustering is not performed in this code, but could be.
  *Inputs from: 'train_GSs' (and, potentially, 'trained_VAE').
  Outputs to: 'Vocabulary_video.mat', 'Vocabulary_odometry.mat', 'Vocabulary_full_structure.mat', 'Clustering_plot.png' in the base dataset folder.*

- `Alg208_ov_ExtractDataFileWithVideoAndParametersSingleTrajectory.py`: prepares the input file to the KVAE training, saving in a single python file the image data and GSs,  and divides them in sub-sequences.
  *Inputs from: 'train_GSs', 'validation_GSs', 'test_GSs', 'train_images', 'validation_images', 'test_images'.
  Outputs to: 'InputsToFinalModel'.*

- `Alg209_ov_TrainKVAE.py`: function for training the KVAE. It starts the training from the learned VAE. The parameters of the KVAE must be inserted in `Config_KVAE.json` under the 'ConfigurationFiles' folder. Note that some are the same of the VAE.
  *Inputs from: 'InputsToFinalModel'.
  Outputs to: 'trained_KVAE'.*

- `Alg210_ov_Final_VideoClusteringExtraction.m`: the video clusters are extracted, keeping the same assignments as in the odometry. 
  Inputs from: 'train_GSs', 'trained_KVAE'.
  *Outputs to: 'Vocabulary_odometry_cut', 'Odometry_based_vocabulary' in the base dataset folder.*

## Testing

Testing proceeds in the following steps:

- The model is tested with multiple MJPF parameters on the validation dataset;

- The parameters giving the best localization result are selected;

- The model is tested on the training dataset with the selected parameters;

- Anomalies are calculated on the training dataset and mean and standard deviation
  of them are extracted;

- The model is tested again on the validation dataset with the selected parameters
  and with multiple anomaly thresholds;

- The thresholds giving the best result is selected;

- Testing is performed on the test dataset;

- Results and plots are extracted and saved.

The codes performing the above steps are the following:

- Alg300_ov_`TrackingValidationWithMultipleMJPFParameters.py:` performs tracking on the validation data, trying multiple parameters written in 
  `Config_KVAE_combined_multiple_parameters.json`.
  *Inputs from: 'trained_KVAE' + 'Config_KVAE_combined' + 'Config_KVAE_combined_multiple_parameters' + vocabularies.
  Outputs to: 'Validation_multiple_parameters_tests'.*

- `Alg301_ov_ChoosingParametersForTraining.m`: take the parameters that gave best results on the validation data and insert them inside the configuration file `Config_KVAE_combined.json`.
  *Inputs from: 'Validation_multiple_parameters_tests'.
  Outputs to: 'Config_KVAE_combined'.*

- `Alg302_ov_TrackingTrainingWithBestParameters.py`: Perform tracking on the training data with the parameters that were the best on the validation set.
  *Inputs from: 'trained_KVAE' + 'Config_KVAE_combined' + vocabularies.
  Outputs to: 'Tracking_results_training_best_parameters'.*

- `Alg303_FindMeanAndStdOfTrainingAnomalies.m`: Finding the mean and the standard deviation values of each anomaly extracted on the training data.
  *Inputs from: 'Tracking_results_training_best_parameters'.
  Outputs to: 'Config_KVAE_combined'.*

- `Alg304_ov_TrackingValidationWithMultipleThresholds.py`: performs tracking on the validation data, trying multiple threholds and parameters on thresholds.
  *Inputs from: 'trained_KVAE' + 'Config_KVAE_combined' + 'Config_KVAE_combined_multiple_thresholds' + vocabularies.
  Outputs to: 'Validation_multiple_thresholds_tests'.*

- `Alg305_ov_ChoosingParametersForTesting.m`: the parameters that gave best
  results on the validation data and insert them inside the configuration file 'Config_KVAE_combined'.
  *Inputs from: 'Validation_multiple_thresholds_tests'.
  Outputs to: 'Config_KVAE_combined'.*

- `Alg305_ov_TrackingOnTesting.py:` Perform tracking on the testing dataset, with the parameters and thresholds that were the best over the validation dataset.
  *Inputs from: 'trained_KVAE' + 'Config_KVAE_combined' + vocabularies.
  Outputs to: 'Tracking_results'.*

- `Alg306_ov_PrintTrackingResults.m`: Print the results from tracking.

*Note on tracking with multiple parameters*: the code saves the different attempts in a folder with the name of the parameter and the value. Consequently, the name of some parameters have been kept short, as Windows would not allow folders with too long names.

## On-the-fly testing

Demo's on-the-fly testing code still to be re-ordered and added.

## Possible additional improvements

Some elements of the code that should be improved:

- The stopping of the KVAE learning should be performed as for the VAE, i.e., using the validation data, instead of setting a certain number of epochs a priori;

- Final testing on the test dataset could be made to start from several points, as done for the first multiple validation tests;
