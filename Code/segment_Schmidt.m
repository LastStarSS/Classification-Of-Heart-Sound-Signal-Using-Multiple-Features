addpath('Schmidt-Segmentation');

%% Example_Schmidt_script
% A script to demonstrate the use of the Schmidt segmentation algorithm

close all;
clear all;
addpath('C:\Users\yaseen\Desktop\MyFolder\Springer-Segmentation-Code-master\Springer-Segmentation-Code-master');

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
[test_AR,Fs] = audioread('C:\Users\yaseen\Desktop\Lecture02-PCG-segment\data\New_MR_077.wav');
[assigned_states] = runSchmidtSegmentationAlgorithm(test_AR*8, Fs, B_matrix, pi_vector, true);
diff_info = diff(assigned_states);
idx = find(diff_info~=0);

state_info = assigned_states;
heart_data = test_AR;
for i=1:length(idx),
    if (i==1),
        st = 1;
        ed = idx(i);
    else
        st = idx(i-1)+1;
        ed = idx(i);
    end
    
    seg_info = assigned_states(st:ed);
    if seg_info(1)==1,
        break;
    else
        state_info(st:ed-1) = -1;
        heart_data(st:ed-1) = -1;
    end
end

idx2 = find(state_info==-1);
heart_data(idx2)=[]; heart_data(1)=[];
state_info(idx2)=[]; state_info(1)=[];
figure; plot(heart_data,'k-'); hold on; plot(state_info,'r-');


diff_info = diff(state_info);
idx3 = find(diff_info~=0);

HeartBeats = struct(); 
cnt1=0;cnt2=0;cnt3=0;cnt4=0;
for i=1:length(idx3),
    if (i==1),
        st = 1;
        ed = idx3(i);
    else
        st = idx3(i-1)+1;
        ed = idx3(i);
    end
    
    seg_info = state_info(st:ed);
    
    if seg_info(1)==1 && cnt1==0,
        S1 = heart_data(st:ed);
        cnt1=1;
    elseif seg_info(2)==2  && cnt2==0,
        ST = heart_data(st:ed);
        cnt2=1;
    elseif seg_info(3)==3  && cnt3==0,
        S2 = heart_data(st:ed);
        cnt3=1;
    elseif seg_info(4)==4  && cnt4==0,
        DT = heart_data(st:ed);
        cnt4=1;
    end
    
    if cnt1+cnt2+cnt3+cnt4==4,
        break;
    end
end

figure; 
plot(1:length(S1),S1,'k-'); hold on;
plot(length(S1)+1:length(S1)+length(ST),ST,'r-'); hold on;
plot(length(S1)+length(ST)+1:length(S1)+length(ST)+length(S2),S2,'g-'); hold on;
plot(length(S1)+length(ST)+length(S2)+1:length(S1)+length(ST)+length(S2)+length(DT),DT,'b-');

FirstHeart = [S1; ST; S2; DT];
audiowrite('AR_002_firstheart.wav',FirstHeart,Fs);


