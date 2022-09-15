function [a_states, z_states, alphas, newClustersSequence,...
    videoVocabulary, A_matrices, B_matrices] = ...
    LoadOutputsFromKVAETraining(paths)

[a_states, ~] = loadObjectGivenFileName(...
    paths.path_a_states_KVAE_training);
[z_states, ~] = loadObjectGivenFileName(...
    paths.path_z_states_KVAE_training);
[alphas, ~] = loadObjectGivenFileName(...
    paths.path_KVAE_alphas);
[newClustersSequence, ~] = loadObjectGivenFileName(...
    paths.path_to_cut_cluster_sequence);
newClustersSequence = newClustersSequence + 1;
[videoVocabulary, ~] = loadObjectGivenFileName(...
    paths.path_to_cluster_vocabulary_video_new);
[A_matrices, ~] = loadObjectGivenFileName(...
    paths.path_A_matrices_train);
[B_matrices, ~] = loadObjectGivenFileName(...
    paths.path_B_matrices_train);

end