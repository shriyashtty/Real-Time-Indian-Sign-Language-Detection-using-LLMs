import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("Loading dataset...")
df = pd.read_csv('hand_data.csv', header=None)
X_raw = df.iloc[:, :-1].values 
y_raw = df.iloc[:, -1].values  

print("Applying Emergency Rotation Augmentation...")
X_augmented = []
y_augmented = []

# Loop through every single row in your CSV
for idx, row in enumerate(X_raw):
    # 1. Keep the perfectly upright original hand
    X_augmented.append(row)
    y_augmented.append(y_raw[idx])
    
    # 2. Synthetically tilt the hand by -30, -15, +15, and +30 degrees
    for angle in [-30, -15, 15, 30]:
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        new_row = []
        # Process the 84 coordinates in (x, y) pairs
        for i in range(0, 84, 2):
            nx, ny = row[i], row[i+1]
            
            # If this is the missing second hand (padded with 0s), leave it as 0
            if nx == 0.0 and ny == 0.0:
                new_row.extend([0.0, 0.0])
            else:
                # Apply the 2D rotation matrix
                new_row.extend([nx * cos_a - ny * sin_a, nx * sin_a + ny * cos_a])
                
        X_augmented.append(new_row)
        y_augmented.append(y_raw[idx])

# Replace the old data with our massive new dataset
X = np.array(X_augmented)
y = np.array(y_augmented)
print(f"Dataset multiplied 5x! Now training on {len(X)} hand samples.")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(np.unique(y))

with open('classes.txt', 'w') as f:
    for class_name in encoder.classes_:
        f.write(f"{class_name}\n")

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("\nBuilding the 84-Coordinate Neural Network...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(84,)), 
    Dropout(0.2), 
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\nStarting training...")
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

model.save('isl_coordinate_model.h5')
print("\nSuccess! Tilt-Proof Model saved as 'isl_coordinate_model.h5'.")

from sklearn.metrics import classification_report
import numpy as np

print("\n--- GENERATING IEEE METRICS REPORT ---")
# 1. Ask the model to predict the test dataset
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# 2. Print the full classification report
report = classification_report(y_test, y_pred_classes, target_names=encoder.classes_)
print(report)