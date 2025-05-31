import os
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Paths
metadata_path = "new-dataset/metadata.csv"
dataset_path = "new-dataset"
output_path = "spectrograms"
os.makedirs(output_path, exist_ok=True)

# Read metadata
df = pd.read_csv(metadata_path)

# Parameters
sr = 22050
duration = 4  # seconds
img_size = (128, 128)

# Create spectrograms
for index, row in df.iterrows():
    emotion = row["emotion"]
    filename = row["filepath"]
    full_path = os.path.join(dataset_path, filename)

    try:
        y, sr = librosa.load(full_path, duration=duration)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Path to save image
        emotion_folder = os.path.join(output_path, emotion)
        os.makedirs(emotion_folder, exist_ok=True)
        save_path = os.path.join(emotion_folder, os.path.splitext(os.path.basename(filename))[0] + ".png")

        # Save spectrogram image
        plt.figure(figsize=(2, 2))
        librosa.display.specshow(mel_spec_db, sr=sr)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Failed to process {filename}: {e}")
