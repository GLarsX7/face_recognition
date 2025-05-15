# Face Recognition App (Tkinter + MediaPipe)

A simple facial recognition desktop application using real-time webcam input, built with Python, OpenCV, MediaPipe, and Tkinter.  
Users can register their face and perform face login with lightweight facial feature encoding and matching.

## Features

- 📸 Real-time face detection using MediaPipe FaceMesh.
- 🧠 Facial encoding with alignment and normalization for robust recognition.
- 👤 User registration and face login with Euclidean + cosine similarity matching.
- 💾 Local data storage (`face_data.json`).
- 🖼️ GUI built with Tkinter and live video feed.

## Demo

<img src="https://user-images.githubusercontent.com/your-demo-image.png" width="500"/>

## Requirements

- Python 3.10
- Webcam

Install required packages:

```bash
pip install -r requirements.txt
