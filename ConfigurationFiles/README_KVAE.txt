
Configuration values to define:

- config_Filename: name of the file where to save the JSON/txt configuration file. Do not put the extension here;

- dimension_x and dimension_y: dimensions to which to reduce the image to;
- image_channels: number of image channels (code has been tested only with image_channels = 3, i.e., RGB);
- sequence_length: length of the sequences for backprop.

- z_meaning: # 0 = odometry, 1 = theta+velNorm, 2 = distance. For now only = 0 is active. Don't inser the other ones.

- filter_size: kernel size of VAE filters;
- dim_filters: channels of VAE filters (e.g., "16,32,64,128");
- stride: stride of VAE filters;

- VAE_already_trained: (true or false);
- trained_VAE_file_name: name of trained VAE. This is taken from the 'output_folder'.
- KVAE_only_kf_already_trained: KVAE trained up to kf training included (true or false);
- trained_KVAE_file_name: name of trained KVAE. This is taken from the 'output_folder'.

- dim_a: dimension of first latent state;
- dim_z: dimension of second latent state;

- noise_emission: noise level for the measurement noise matrix;
- noise_transition: noise level for the process noise matrix;
- init_cov: variance of the initial state;

- skewness: skewness at beginning;
- skew_increase_per_epoch: increase of skew per epoch;

- batch_size: batch size for KVAE training;
- batch_size_test: batch size for KVAE testing;
- batch_size_VAE: batch size for VAE training (not used in the current version, where VAE is trained before);

- lr_only_vae: learning rate when training the VAE (not used in the current version, where VAE is trained before);
- init_lr: initial learning rate when training the KVAE;
- init_lrVAE: initial learning rate when training the KVAE with VAE together;

- max_grad_norm_kf, max_grad_norm_VAE: value at which gradient cutting is performed;
- init_kf_matrices: standard deviation of noise used to initialize B and C;

- only_vae_epochs: number of epochs of VAE training; 
- kf_update_steps: number of epochs of training of the KF alone;
- train_epochs: TOTAL number of training epochs;

- decay_rate;
- decay_steps;
- weight_decay;

- alpha_VAEonly_training: weight for alpha-VAE when training the VAE alone. 
  This is the value of KLD against reconstruction (not used in the current version, where VAE is trained before);
- KLDLoss_weight: weight for KLD part when training the KVAE;
- RecLoss_weight: weight of reconstruction loss when training the KVAE;
- transitionLoss_weight: weight of transition loss when training the KVAE;
- emissionLoss_weight: weight of emission loss when training the KVAE;
- dLoss_weight: weight of distance from odometry loss when training the KVAE (for KVAE_D only);

- deleteThreshClusterSmoothing: if an assignment to a cluster lasts less than this value, 
  do not accept it. This smoothens the cluster assignments;

- minFile, maxFile: files where min and max of GSs of odometry are saved. 
- minFile, maxFile, training_data_file, validation_data_file, testing_data_file, clustering_data_file, output_folder:
  files indicated w.r.t. the BaseDataFolder.






