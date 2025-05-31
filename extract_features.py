import os
import numpy as np
import librosa
import pickle

# ✅ Path to your RAVDESS dataset folder
data_path = "ravdess-data"
features = []
labels = []

# Map RAVDESS emotion codes to actual labels
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=3.0, sr=22050*2, offset=0.3)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

print("🔁 Extracting features...")
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            try:
                emotion_code = file.split("-")[2]
                emotion_label = emotion_map.get(emotion_code)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(emotion_label)
            except Exception as e:
                print(f"❌ Skipping file {file}: {e}")

print(f"✅ Extracted features for {len(features)} audio files")

# 🔒 Save features and labels
with open("features.npy", "wb") as f:
    np.save(f, np.array(features))

with open("labels.npy", "wb") as f:
    np.save(f, np.array(labels))

print("💾 Features and labels saved successfully!")
