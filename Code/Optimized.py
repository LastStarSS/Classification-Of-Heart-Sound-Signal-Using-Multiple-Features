import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pywt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import layers, models


# 1. Settings

DATASET_PATH = "D:/Coding/NCKH/Classification-Of-Heart-Sound-Signal-Using-Multiple-Features/Data"
LABELS = ["MR", "MS", "N", "MVP"]
N_MFCC = 16
N_FRAMES = 30
IMG_CHANNELS = 2
MODEL_PATH = "saved_model/heart_model.h5"


# 2. Load dataset and extract MFCC + DWT features

def load_dataset(dataset_path=DATASET_PATH, labels=LABELS, n_mfcc=N_MFCC, n_frames=N_FRAMES):
    features = []
    file_labels = []
    for label in labels:
        folder = os.path.join(dataset_path, label)
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                filepath = os.path.join(folder, file)
                y, sr = librosa.load(filepath, sr=None)

                # MFCC
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                if mfcc.shape[1] < n_frames:
                    continue
                mfcc_feat = mfcc[:, :n_frames]

                # DWT
                coeffs = pywt.wavedec(y, 'db4', level=4)
                dwt_feat = coeffs[0]
                if len(dwt_feat) < n_frames:
                    dwt_feat = np.pad(dwt_feat, (0, n_frames - len(dwt_feat)))
                else:
                    dwt_feat = dwt_feat[:n_frames]
                dwt_feat = dwt_feat.reshape(1, n_frames)
                dwt_feat_rep = np.repeat(dwt_feat, n_mfcc, axis=0)

                # Stack MFCC + DWT as channels
                combined_feat = np.stack([mfcc_feat, dwt_feat_rep], axis=-1)
                features.append(combined_feat)
                file_labels.append(label)

    X = np.array(features)
    y = np.array(file_labels)

    # One-hot encode labels
    encoder = LabelBinarizer()
    y_onehot = encoder.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test, encoder


# 3. Build CNN model

def build_cnn(input_shape=(N_MFCC, N_FRAMES, IMG_CHANNELS), n_classes=len(LABELS)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 4. Train and save model
def train_model(model, X_train, y_train, X_test, y_test,
                model_path=MODEL_PATH, epochs=30, batch_size=32,
                acc_plot_path="saved_model/accuracy_plot.png",
                loss_plot_path="saved_model/loss_plot.png"):
    
    if not os.path.exists(model_path):
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test)
        )

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Model saved at {model_path}")

        # Accuracy plot
        plt.figure(figsize=(8,5))
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.legend()
        plt.title("Training vs Validation Accuracy")
        plt.tight_layout()
        plt.savefig(acc_plot_path)
        print(f"Accuracy plot saved at {acc_plot_path}")
        plt.show()

        # Loss plot
        plt.figure(figsize=(8,5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.ylim(0, 1)
        plt.legend()
        plt.title("Training vs Validation Loss")
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        print(f"Loss plot saved at {loss_plot_path}")
        plt.show()

    else:
        print(f"Model already exists at {model_path}")

        # Display saved plots
        for plot_path in [acc_plot_path, loss_plot_path]:
            if os.path.exists(plot_path):
                img = plt.imread(plot_path)
                plt.figure(figsize=(8,5))
                plt.imshow(img)
                plt.axis('off')
                plt.show()


# 5. Test random samples with CWT visualization
def test_random_samples(n=10, model_path=MODEL_PATH):
    model = models.load_model(model_path)

    # Collect all test files
    test_files = []
    for label in LABELS:
        folder = os.path.join(DATASET_PATH, label)
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                test_files.append((os.path.join(folder, file), label))

    random_files = random.sample(test_files, n)

    plt.figure(figsize=(15, 10))
    for idx, (filepath, true_label) in enumerate(random_files, 1):
        y_audio, sr = librosa.load(filepath, sr=None)

        # MFCC + DWT for prediction
        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=N_MFCC)
        if mfcc.shape[1] < N_FRAMES:
            continue
        mfcc_feat = mfcc[:, :N_FRAMES]

        coeffs = pywt.wavedec(y_audio, 'db4', level=4)
        dwt_feat = coeffs[0]
        if len(dwt_feat) < N_FRAMES:
            dwt_feat = np.pad(dwt_feat, (0, N_FRAMES - len(dwt_feat)))
        else:
            dwt_feat = dwt_feat[:N_FRAMES]
        dwt_feat = dwt_feat.reshape(1, N_FRAMES)
        dwt_feat_rep = np.repeat(dwt_feat, N_MFCC, axis=0)

        combined_feat = np.stack([mfcc_feat, dwt_feat_rep], axis=-1)
        combined_feat = combined_feat[np.newaxis, ...]  # add batch dim

        pred_prob = model.predict(combined_feat)
        pred_label = encoder.classes_[np.argmax(pred_prob)]

        # CWT visualization
        coeffs_cwt, freqs = pywt.cwt(y_audio, scales=np.arange(1,128), wavelet='morl')
        plt.subplot(2, 5, idx)
        plt.imshow(np.abs(coeffs_cwt), aspect='auto', cmap='magma', origin='lower')
        plt.title(f"T: {true_label}\nP: {pred_label}", fontsize=10)
        plt.xlabel("Time")
        plt.ylabel("Scale")
        plt.tight_layout()
    plt.show()


# 6. Execute all
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, encoder = load_dataset()
    model = build_cnn()
    train_model(model, X_train, y_train, X_test, y_test)
    test_random_samples(10)
