import os
import librosa
import pywt
import numpy as np
import pandas as pd
from scipy.io import wavfile

def extract_dwt_features(signal, wavelet='db4', level=4, length=352):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    feat = np.concatenate([c for c in coeffs])
    return np.pad(feat, (0, max(0, length-len(feat))))[:length]

root = "Heart_Beat_sound_signal_3Periods200files"
features = []
labels = []

for subdir, _, files in os.walk(root):
    for file in files:
        if file.lower().endswith(".wav"):
            fs, y = wavfile.read(os.path.join(subdir, file))
            y = y.astype(float)

            # DWT part
            dwt_feat = extract_dwt_features(y, length=352)

            # MFCC part (16 coeffs × 30 frames = 480)
            y_lib, sr = librosa.load(os.path.join(subdir, file), sr=None)
            mfcc = librosa.feature.mfcc(y=y_lib, sr=sr, n_mfcc=16)
            
            if mfcc.shape[1] >= 30:
                mfcc_feat = mfcc[:, :30].flatten()[:480]
                combined = np.concatenate([dwt_feat, mfcc_feat])  # 352+480=832
                features.append(combined)
                labels.append(os.path.basename(subdir))

# Save
df = pd.DataFrame(features)
df["label"] = labels
df.to_csv("dwt_mfcc_features.csv", index=False)
