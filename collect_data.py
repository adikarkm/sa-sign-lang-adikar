import os
import cv2
import csv
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'hand_landmarker.task')
CSV_FILE = os.path.join(BASE_DIR, 'hand_landmarks.csv')

# Expanded classes list
CLASSES = ["african beer", "hello", "how", "how are you", "okay", "think", "thank you", "please", "yes", "no", "name"]
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
FRAME_STRIDE = 5 

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading Hand Landmarker model asset...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Download complete!")

def process_landmarks(results, class_name, csv_writer):
    if not results.hand_landmarks:
        return False

    row_original = [class_name]
    row_inverted = [class_name]
    
    # Process up to 2 hands
    hands_to_process = results.hand_landmarks[:2]
    
    for hand_landmarks in hands_to_process:
        wrist = hand_landmarks[0] 
        wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
        
        for landmark in hand_landmarks:
            relative_x = landmark.x - wrist_x
            relative_y = landmark.y - wrist_y
            relative_z = landmark.z - wrist_z
            
            row_original.extend([relative_x, relative_y, relative_z])
            row_inverted.extend([relative_x * -1, relative_y, relative_z])
    
    # Zero-padding for single-hand detections
    if len(hands_to_process) == 1:
        row_original.extend([0] * 63)
        row_inverted.extend([0] * 63)
        
    if len(row_original) == 127:
        csv_writer.writerow(row_original)
        csv_writer.writerow(row_inverted)
        return True
    return False

def collect_data():
    download_model()
    
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE 
    )
    
    print("Initializing MediaPipe Hand Landmarker...")
    detector = vision.HandLandmarker.create_from_options(options)

    # Initialize CSV header if not exists, or open in append mode
    file_exists = os.path.exists(CSV_FILE)
    mode = 'a' if file_exists else 'w'
    
    with open(CSV_FILE, mode=mode, newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        if not file_exists:
            header = ['label']
            for i in range(42):
                header.extend([f'x{i}', f'y{i}', f'z{i}'])
            csv_writer.writerow(header)

        for class_name in CLASSES:
            folder_path = os.path.join(BASE_DIR, class_name)
            if not os.path.exists(folder_path):
                print(f"Directory not found (Skipped) -> {folder_path}")
                continue

            print(f"Processing folder: {class_name}...")
            vectors_extracted = 0
            files_skipped = 0
            
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                if file_name.lower().endswith(VIDEO_EXTENSIONS):
                    cap = cv2.VideoCapture(file_path)
                    frame_idx = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if frame_idx % FRAME_STRIDE == 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                            results = detector.detect(mp_image)
                            if process_landmarks(results, class_name, csv_writer):
                                vectors_extracted += 2
                        frame_idx += 1
                    cap.release()
                else:
                    image = cv2.imread(file_path)
                    if image is None:
                        files_skipped += 1
                        continue

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                    results = detector.detect(mp_image)
                    if process_landmarks(results, class_name, csv_writer):
                        vectors_extracted += 2
                    else:
                        files_skipped += 1

            print(f"Class '{class_name}': Extracted {vectors_extracted} vectors. Skipped: {files_skipped}")

    print("Data collection and extraction complete.")

if __name__ == "__main__":
    collect_data()
