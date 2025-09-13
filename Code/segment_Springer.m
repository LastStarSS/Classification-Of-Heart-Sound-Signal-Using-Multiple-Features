addpath('Springer-Segmentation');

%%
close all;
clear all;

%% Load the default options:
% These options control options such as the original sampling frequency of
% the data, the sampling frequency for the derived features and whether the
% mex code should be used for the Viterbi decoding:
springer_options = default_Springer_HSMM_options;

%% Load the audio data and the annotations:
% These are 6 example PCG recordings, downsampled to 1000 Hz, with
% annotations of the R-peak and end-T-wave positions.
load('example_data.mat');

%% Split the data into train and test sets:
% Select the first 5 recordings for training and the sixth for testing:
train_recordings = example_data.example_audio_data([1:6]);
train_annotations = example_data.example_annotations([1:6],:);


%% Train the HMM:
[B_matrix, pi_vector, total_obs_distribution] = trainSpringerSegmentationAlgorithm(train_recordings,train_annotations,springer_options.audio_Fs, false);

[test_AR,Fs] = audioread('../data/AR_002.mp3');
[assigned_states] = runSpringerSegmentationAlgorithm(test_AR, Fs, B_matrix, pi_vector, total_obs_distribution, true);


[test_N,Fs] = audioread('../data/N_004.mp3');
[assigned_states] = runSpringerSegmentationAlgorithm(test_N(:,1)*8, Fs/2, B_matrix, pi_vector, total_obs_distribution, true);
