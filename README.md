---
title: Sign Language Recognition App
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🤟 Sign Language Recognition Web App

Real-time Sign Language Gesture Detection powered by MediaPipe Hand Landmarker and Keras Deep Learning Model.

## Supported Gestures
- `african beer`
- `hello`
- `how`
- `how are you`
- `okay`
- `think`

## Deployment Instructions

### Option 1: Deploy on Hugging Face Spaces (Free Cloud Hosting)
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose a Space Name (e.g. `sign-language-app`).
3. Select **Gradio** as the Space SDK.
4. Clone your Space repository locally or upload these files (`app.py`, `landmark_model.keras`, `hand_landmarker.task`, `requirements.txt`, `README.md`).
5. Your web app will automatically build and go live at `https://huggingface.co/spaces/<your-username>/<your-space-name>`.

### Option 2: Run Locally
```bash
pip install -r requirements.txt
python app.py
```
Open `http://localhost:7860` in your web browser.

### Option 3: Host Locally with ngrok Public Tunnel
```bash
python run_local_tunnel.py
```
This generates a shareable public HTTPS web link to your local app.
