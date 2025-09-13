addpath('Schmidt-Segmentation');

%% Example_Schmidt_script
% A script to demonstrate the use of the Schmidt segmentation algorithm

close all;
clear all;

%% Load the default options:
% These options control options such as the original sampling frequency of
% the data, the sampling frequency for the derived features and whether the
% mex code should be used for the Viterbi decoding:
schmidt_options = default_Schmidt_HSMM_options;

%% Load the audio data and the annotations:
% These are 6 example PCG recordings, downsampled to 1000 Hz, with
% annotations of the R-peak and end-T-wave positions.
load('example_data.mat');

%% Split the data into train and test sets:
% Select the first 5 recordings for training and the sixth for testing:
train_recordings = example_data.example_audio_data([1:6]);
train_annotations = example_data.example_annotations([1:6],:);


%% Train the HMM:
[B_matrix, pi_vector] = trainSchmidtSegmentationAlgorithm(train_recordings,train_annotations,schmidt_options.audio_Fs, false);


%% Run the HMM on an unseen test recording:
[test_AR,Fs] = audioread('../data/AR_002.mp3');
[assigned_states] = runSchmidtSegmentationAlgorithm(test_AR*8, Fs, B_matrix, pi_vector, true);
diff_info = diff(assigned_states);
idx = find(diff_info~=0);

HeartBeats = struct(); cnt1=1;
isFirst = 0;
for i=1:length(idx),
    
    if (i==1),
        st = 1;
        ed = idx(i);
    else
        st = idx(i-1);
        ed = idx(i);
    end
    
    seg_info = assigned_states(st:ed-1);
    
    if seg_info(1)==1,
        HeartBeats(cnt1).S1 = test_AR(st:ed-1);
    elseif seg_info(1)==2,
        HeartBeats(cnt1).ST = test_AR(st:ed-1);
    elseif seg_info(1)==3,
        HeartBeats(cnt1).S2 = test_AR(st:ed-1);
    elseif seg_info(1)==4,
        HeartBeats(cnt1).DT = test_AR(st:ed-1);
    end
    
end



[test_N,Fs2] = audioread('../data/N_004.mp3');
[assigned_states] = runSchmidtSegmentationAlgorithm(test_N(:,2)*8, Fs/2, B_matrix, pi_vector, true);

