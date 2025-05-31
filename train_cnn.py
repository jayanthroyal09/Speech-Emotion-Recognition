import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import cv2

# Directory where spectrogram images are saved (each subfolder = label)
image_dir = os.path.join("C:", os.sep, "Users", "dsahi", "Speech-Emotion-Analyzer", "spectrograms")
img_size = 128

# Load and preprocess images
images = []
labels = []

print("🖼️ Scanning for spectrogram images in all subfolders...")
for folder in os.listdir(image_dir):
    folder_path = os.path.join(image_dir, folder)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(folder)  # Folder name = emotion label

if not images or not labels:
    raise ValueError("❌ No images or labels found. Check the spectrograms folder path and structure.")

images = np.array(images) / 255.0  # Normalize pixel values
labels = np.array(labels)

# Encode labels (convert strings to integers)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = tf.keras.utils.to_categorical(labels_encoded)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Build CNN model
print("🧠 Building CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(labels)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
print("🏋️ Training the model...")
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Save the model and label classes
model.save("cnn_model.keras")  # Save in native Keras format
np.save("label_classes.npy", le.classes_)

print("✅ CNN model trained and saved successfully!")

# -------- NEW: Evaluation Metrics --------
print("\n📊 Evaluating model performance on test set...")
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print accuracy
accuracy = accuracy_score(y_true, y_pred_labels)
print("✅ Test Accuracy:", round(accuracy * 100, 2), "%")

# Print classification report
print("\n📋 Classification Report:\n")
print(classification_report(y_true, y_pred_labels, target_names=le.classes_))
