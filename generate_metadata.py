import os
import pandas as pd

base_dir = r"C:\Users\dsahi\Speech-Emotion-Analyzer\new-dataset"
data = []

# Mapping for different datasets
crema_emotions = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fearful",
    "HAP": "happy", "NEU": "calm", "SAD": "sad"
}

ravdess_emotions = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

savee_emotions = {
    "a": "angry", "d": "disgust", "f": "fearful", "h": "happy",
    "n": "calm", "sa": "sad", "su": "surprised"
}

# CREMA
crema_path = os.path.join(base_dir, "Crema")
for file in os.listdir(crema_path):
    if file.endswith(".wav"):
        emotion = crema_emotions.get(file.split("_")[2])
        if emotion:
            data.append([os.path.join(crema_path, file), emotion])

# RAVDESS
ravdess_path = os.path.join(base_dir, "Ravdess", "audio_speech_actors_01-24")
for actor_folder in os.listdir(ravdess_path):
    actor_path = os.path.join(ravdess_path, actor_folder)
    for file in os.listdir(actor_path):
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]
            emotion = ravdess_emotions.get(emotion_code)
            if emotion:
                data.append([os.path.join(actor_path, file), emotion])

# SAVEE
savee_path = os.path.join(base_dir, "Savee")
for file in os.listdir(savee_path):
    if file.endswith(".wav"):
        prefix = file.split("_")[0]
        key = prefix[-2:] if prefix[-2:] in savee_emotions else prefix[-1]
        emotion = savee_emotions.get(key)
        if emotion:
            data.append([os.path.join(savee_path, file), emotion])

# TESS
tess_path = os.path.join(base_dir, "Tess")
for folder in os.listdir(tess_path):
    folder_path = os.path.join(tess_path, folder)
    emotion = folder.split("_")[-1].lower()
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            data.append([os.path.join(folder_path, file), emotion])

# Save to CSV
df = pd.DataFrame(data, columns=["filepath", "emotion"])
csv_path = os.path.join(base_dir, "metadata.csv")
df.to_csv(csv_path, index=False)
print(f"Metadata saved to: {csv_path}")
