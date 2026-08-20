import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "hand_landmarks.csv")
MODEL_PATH = os.path.join(BASE_DIR, "landmark_model.keras")

CLASSES = ["african beer", "hello", "how", "how are you", "okay", "think", "thank you", "please", "yes", "no", "name"]

def train():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"Dataset missing at {CSV_FILE}. Please run collect_data.py first.")

    print("Loading landmark datasets...")
    df = pd.read_csv(CSV_FILE)

    # Filter dataset to match our classes list
    df = df[df['label'].isin(CLASSES)]
    
    X = df.drop('label', axis=1).values
    y_strings = df['label'].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_strings)
    class_names = list(encoder.classes_)
    num_classes = len(class_names)
    
    print(f"Dataset summary: {len(df)} samples across {num_classes} classes.")
    print(f"Class mapping: {dict(enumerate(class_names))}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build sequential deep neural net
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(126,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training neural network...")
    model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32, 
        validation_data=(X_test, y_test), 
        verbose=1
    )

    print(f"Saving compiled model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Model training and saving completed successfully!")

if __name__ == "__main__":
    train()
