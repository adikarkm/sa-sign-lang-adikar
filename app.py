import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import gradio as gr

# Setup base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "landmark_model.keras")
LANDMARKER_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

CLASS_NAMES = ["african beer", "hello", "how", "how are you", "okay", "think"]

# Global model initialization
keras_model = None
detector = None

def load_models():
    global keras_model, detector
    if keras_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        keras_model = tf.keras.models.load_model(MODEL_PATH)

    if detector is None:
        if not os.path.exists(LANDMARKER_PATH):
            # Download if missing
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, LANDMARKER_PATH)

        base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        detector = vision.HandLandmarker.create_from_options(options)

load_models()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process_frame(image):
    if image is None:
        return None, {name: 0.0 for name in CLASS_NAMES}

    # Ensure RGB format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    annotated_image = image.copy()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    
    results = detector.detect(mp_image)
    
    predictions_dict = {name: 0.0 for name in CLASS_NAMES}
    
    if results.hand_landmarks:
        live_coordinates = []
        
        for hand_landmarks in results.hand_landmarks:
            # Convert for drawing
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
            ])
            mp_drawing.draw_landmarks(
                annotated_image, 
                hand_landmarks_proto, 
                mp_hands.HAND_CONNECTIONS
            )
            
            wrist = hand_landmarks[0]
            wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
            
            for lm in hand_landmarks:
                live_coordinates.extend([
                    lm.x - wrist_x,
                    lm.y - wrist_y,
                    lm.z - wrist_z
                ])
                
        # Zero padding for single hand
        if len(results.hand_landmarks) == 1:
            live_coordinates.extend([0] * 63)
            
        if len(live_coordinates) == 126:
            input_data = np.array([live_coordinates])
            preds = keras_model.predict(input_data, verbose=0)[0]
            
            for idx, prob in enumerate(preds):
                predictions_dict[CLASS_NAMES[idx]] = float(prob)
                
            best_idx = np.argmax(preds)
            best_conf = preds[best_idx] * 100
            
            label_text = f"Recognized: {CLASS_NAMES[best_idx].upper()} ({best_conf:.1f}%)"
            
            # Overlay result banner
            h, w, _ = annotated_image.shape
            cv2.rectangle(annotated_image, (10, 10), (min(500, w - 10), 65), (20, 20, 20), -1)
            cv2.putText(
                annotated_image, 
                label_text, 
                (20, 48), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 255, 128), 
                2, 
                cv2.LINE_AA
            )

    return annotated_image, predictions_dict

# Build Gradio UI
with gr.Blocks(title="Sign Language Recognition Web App") as demo:
    gr.Markdown(
        """
        # 🤟 Sign Language Recognition Web App
        Real-time Sign Language Gesture Detection powered by **MediaPipe Hand Landmarker** & **TensorFlow Keras**.
        
        ### 📋 Supported Gestures:
        - `african beer` | `hello` | `how` | `how are you` | `okay` | `think`
        """
    )
    
    with gr.Tab("Live Webcam Stream"):
        with gr.Row():
            with gr.Column():
                webcam_input = gr.Image(
                    sources=["webcam"], 
                    type="numpy", 
                    label="Live Camera Stream",
                    streaming=True
                )
            with gr.Column():
                webcam_output_img = gr.Image(label="Hand Landmark Visualization")
                webcam_output_label = gr.Label(num_top_classes=6, label="Gesture Predictions")
                
        webcam_input.stream(fn=process_frame, inputs=[webcam_input], outputs=[webcam_output_img, webcam_output_label])

    with gr.Tab("Upload Image"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    sources=["upload", "clipboard"], 
                    type="numpy", 
                    label="Upload Sign Language Photo"
                )
                analyze_btn = gr.Button("Recognize Sign", variant="primary")
            with gr.Column():
                image_output_img = gr.Image(label="Detected Landmarks")
                image_output_label = gr.Label(num_top_classes=6, label="Prediction Scores")
                
        analyze_btn.click(fn=process_frame, inputs=[image_input], outputs=[image_output_img, image_output_label])
        image_input.change(fn=process_frame, inputs=[image_input], outputs=[image_output_img, image_output_label])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
