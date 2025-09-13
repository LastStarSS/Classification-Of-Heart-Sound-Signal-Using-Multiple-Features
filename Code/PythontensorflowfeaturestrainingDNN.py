import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow.compat.v1 as tf
import random

tf.disable_v2_behavior()


 
# 1. Load dataset and extract MFCC features
 
data_dir = "Data"   # root directory containing MR, MS, N, MVP folders
labels, features = [], []

for label in ["MR", "MS", "N", "MVP"]:
    folder = os.path.join(data_dir, label)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            filepath = os.path.join(folder, file)
            y, sr = librosa.load(filepath, sr=None)

            # Extract MFCC (16 coeffs × 30 frames = 480 features)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=16)
            if mfcc.shape[1] >= 30:
                feat = mfcc[:, :30].flatten()[:480]
                features.append(feat)
                labels.append(label)

X = np.array(features)
y = np.array(labels)

# One-hot encode labels
lb = LabelBinarizer()
y_onehot = lb.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=42)

 
# 2. Define model
 
n_input = X.shape[1]  # 480
n_classes = len(lb.classes_)  # 4

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

n_nodes_hl1, n_nodes_hl2, n_nodes_hl3 = 500, 500, 500

def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([n_input, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.nn.relu(tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases']))
    l2 = tf.nn.relu(tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases']))
    l3 = tf.nn.relu(tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases']))
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    return output

 
# 3. Train, save, and visualize
 
def train_and_save_model(save_path="saved_model/model.ckpt"):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    hm_epochs = 50
    batch_size = 32

    saver = tf.train.Saver()
    train_acc_history, val_acc_history = [], []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_x = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
            acc_train = sess.run(accuracy, feed_dict={x: X_train, y: y_train})
            acc_val = sess.run(accuracy, feed_dict={x: X_test, y: y_test})
            train_acc_history.append(acc_train)
            val_acc_history.append(acc_val)
            print(f"Epoch {epoch+1}/{hm_epochs} Loss: {epoch_loss:.3f} TrainAcc: {acc_train:.3f} ValAcc: {acc_val:.3f}")

        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        saver.save(sess, save_path)
        print(f"Model saved at {save_path}")

        # Confusion matrix
        preds = sess.run(tf.argmax(prediction, 1), feed_dict={x: X_test})
        true_labels = np.argmax(y_test, axis=1)
        cm = confusion_matrix(true_labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lb.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

        # Accuracy curves
        plt.plot(train_acc_history, label="Train Acc")
        plt.plot(val_acc_history, label="Validation Acc")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Training vs Validation Accuracy")
        plt.show()

 
# 4. Random test visualization
 
def test_random_samples(n=10, model_path="saved_model/model.ckpt"):
    prediction = neural_network_model(x)
    saver = tf.train.Saver()

    # Collect all test files with labels
    test_files = []
    for label in ["MR", "MS", "N", "MVP"]:
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                test_files.append((os.path.join(folder, file), label))

    # Pick random subset
    random_files = random.sample(test_files, n)

    with tf.Session() as sess:
        saver.restore(sess, model_path)

        plt.figure(figsize=(15, 10))
        for idx, (filepath, true_label) in enumerate(random_files, 1):
            # Extract MFCC
            y_audio, sr = librosa.load(filepath, sr=None)
            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=16)
            if mfcc.shape[1] >= 30:
                feat = mfcc[:, :30].flatten()[:480].reshape(1, -1)

                # Predict
                pred_class = sess.run(tf.argmax(prediction, 1), feed_dict={x: feat})
                pred_label = lb.classes_[pred_class[0]]

                # Plot MFCC spectrogram
                plt.subplot(2, 5, idx)
                librosa.display.specshow(mfcc, x_axis="time", sr=sr)
                plt.colorbar(format="%+2.0f dB")
                plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
                plt.tight_layout()
        plt.show()

# Run training
# train_and_save_model()
test_random_samples(10)
