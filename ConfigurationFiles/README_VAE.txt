
Configuration values to define:

- config_Filename: name of the file where to save the JSON/txt configuration file. Do not put the extension here;

- dimension_x and dimension_y: dimensions to which to reduce the image to;
- image_channels: number of image channels (code has been tested only with image_channels = 3, i.e., RGB);

- filter_size: kernel size of VAE filters;
- dim_filters: channels of VAE filters (e.g., "16,32,64,128");
- stride: stride of VAE filters;
- VAE_already_trained: (true or false);
- trained_VAE_file_name: name of trained VAE. This is taken from the 'output_folder'.
- dim_a: dimension of VAE latent state;

- batch_size_VAEbatch size for VAE training and validation;
- lr_only_vae: learning rate when training the VAE;
- max_grad_norm_VAEvalue at which gradient cutting is performed;
- only_vae_epochs: number of epochs of VAE training; 

- decay_rate;
- decay_steps;
- weight_decay;

- alpha_VAEonly_training: weight for alpha-VAE when training the VAE alone. 
  This is the value of KLD against reconstruction;

- training_data_file, validation_data_file,test_data_file, output_folder: files indicated w.r.t. the BaseDataFolder.






