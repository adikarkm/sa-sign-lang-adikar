import os
import io
import csv
import json
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKER_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
CSV_FILE = os.path.join(BASE_DIR, "hand_landmarks.csv")
MODEL_PATH = os.path.join(BASE_DIR, "landmark_model.keras")
CLASSES_JSON_PATH = os.path.join(BASE_DIR, "classes.json")

def build_dataset_and_train():
    print("--- STEP 1: INITIALIZING MEDIAPIPE ---")
    base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # 1. Read existing CSV if exists to keep custom words
    existing_rows = []
    if os.path.exists(CSV_FILE):
        print(f"Reading existing data from {CSV_FILE}...")
        df_old = pd.read_csv(CSV_FILE)
        # Filter out previous single letter entries if any to re-generate cleanly
        alphabet_set = {chr(c) for c in range(ord('A'), ord('Z')+1)}
        df_kept = df_old[~df_old['label'].isin(alphabet_set)]
        print(f"Preserving existing word classes: {df_kept['label'].value_counts().to_dict()}")
        for _, row in df_kept.iterrows():
            existing_rows.append(row.tolist())

    # 2. Download ASL alphabet parquet from Hugging Face
    print("--- STEP 2: DOWNLOADING ASL ALPHABET DATASET ---")
    parquet_path = hf_hub_download(
        repo_id='Marxulia/asl_sign_languages_alphabets_v03',
        filename='data/train-00000-of-00001.parquet',
        repo_type='dataset'
    )
    df_asl = pd.read_parquet(parquet_path)
    print(f"Total ASL images available: {len(df_asl)}")

    # Mapping label 0..25 to 'A'..'Z'
    label_to_letter = {i: chr(ord('A') + i) for i in range(26)}

    new_rows = []
    samples_per_letter = 45 # 45 images * 2 (augmented) = 90 vectors per letter

    print("--- STEP 3: EXTRACTING LANDMARKS FOR ALPHABET A-Z ---")
    for label_id, letter in label_to_letter.items():
        subset = df_asl[df_asl['label'] == label_id]
        collected = 0
        
        for _, row in subset.iterrows():
            if collected >= samples_per_letter:
                break
            try:
                img_bytes = row['image']['bytes']
                pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                np_img = np.array(pil_img)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)
                results = detector.detect(mp_img)
                
                if results.hand_landmarks:
                    hands_to_process = results.hand_landmarks[:2]
                    row_orig = [letter]
                    row_flip = [letter]
                    
                    for hand_landmarks in hands_to_process:
                        wrist = hand_landmarks[0]
                        wx, wy, wz = wrist.x, wrist.y, wrist.z
                        for lm in hand_landmarks:
                            rx = lm.x - wx
                            ry = lm.y - wy
                            rz = lm.z - wz
                            row_orig.extend([rx, ry, rz])
                            row_flip.extend([rx * -1, ry, rz])
                            
                    if len(hands_to_process) == 1:
                        row_orig.extend([0] * 63)
                        row_flip.extend([0] * 63)
                        
                    if len(row_orig) == 127:
                        new_rows.append(row_orig)
                        new_rows.append(row_flip)
                        collected += 1
            except Exception as e:
                continue
                
        print(f"Extracted {collected * 2} landmark samples for letter: {letter}")

    # Combine existing words + new alphabet
    all_data = existing_rows + new_rows
    print(f"Total dataset vectors: {len(all_data)}")

    # Write combined CSV
    header = ['label']
    for i in range(42):
        header.extend([f'x{i}', f'y{i}', f'z{i}'])

    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_data)
    print(f"Saved consolidated dataset to {CSV_FILE}")

    # 3. Train Upgraded Neural Network
    print("--- STEP 4: TRAINING DEEP NEURAL NETWORK ---")
    df = pd.read_csv(CSV_FILE)
    X = df.drop('label', axis=1).values
    y_raw = df['label'].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    class_names = [str(c) for c in encoder.classes_]
    num_classes = len(class_names)
    print(f"Total classes ({num_classes}): {class_names}")

    # Save classes.json
    with open(CLASSES_JSON_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved classes.json mapping with {num_classes} classes.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(126,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=80,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Model Test Accuracy: {test_acc * 100:.2f}%")

    model.save(MODEL_PATH)
    print(f"Saved retrained model to {MODEL_PATH}")

if __name__ == "__main__":
    build_dataset_and_train()
