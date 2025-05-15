import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import json
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
FACE_DATA_FILE = "face_data.json"

class FaceApp:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)

        # Show a welcome message
        messagebox.showinfo("Welcome", "Face Recognition System started.")

        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open the camera.")

        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        btn_frame = tk.Frame(window)
        btn_frame.pack(pady=10)

        self.btn_register = tk.Button(btn_frame, text="Register Face", width=20, command=self.start_registration)
        self.btn_register.grid(row=0, column=0, padx=5)

        self.btn_login = tk.Button(btn_frame, text="Face Login", width=20, command=self.start_login)
        self.btn_login.grid(row=0, column=1, padx=5)

        self.btn_snapshot = tk.Button(btn_frame, text="Capture Photo", width=20, command=self.capture_action)
        self.btn_snapshot.grid_forget()  # Initially hidden

        self.btn_list_users = tk.Button(btn_frame, text="List Users", width=20, command=self.list_users)
        self.btn_list_users.grid(row=0, column=3, padx=5)

        self.current_mode = None
        self.username = None

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False  # Better for video input
        )

        self.delay = 15
        self.update()
        self.window.mainloop()

    def start_registration(self):
        # Prompt username for face registration
        self.username = simpledialog.askstring("Registration", "Enter your username:")
        if self.username:
            self.current_mode = 'register'
            self.btn_snapshot.grid(row=0, column=2, padx=5)

    def start_login(self):
        # Start face login mode
        self.current_mode = 'login'
        self.btn_snapshot.grid(row=0, column=2, padx=5)

    def list_users(self):
        # List all registered usernames in a message box
        users = self.load_users()
        if users:
            user_list = "\n".join(users.keys())
            messagebox.showinfo("Registered Users", user_list)
        else:
            messagebox.showinfo("Registered Users", "No users registered.")

    def capture_action(self):
        # Decide action based on current mode
        if self.current_mode == 'register':
            self.register_face()
        elif self.current_mode == 'login':
            self.verify_login()
        self.cleanup()

    def register_face(self):
        # Get face landmarks from current frame
        encodings = self.process_frame(draw=False)
        if not encodings:
            messagebox.showwarning("Warning", "No face detected. Try again.")
            return

        samples = []
        # Capture 3 samples with slight delay to get varied angles
        for _ in range(3):
            time.sleep(0.5)
            enc = self.process_frame(draw=False)
            if enc:
                normalized = self.normalize_and_align(enc[0])
                samples.append(normalized)

        # Calculate average encoding
        mean_encoding = np.mean(samples, axis=0)
        self.save_data("users", self.username, mean_encoding.tolist())
        messagebox.showinfo("Success", f"Face registered for {self.username}!")

    def verify_login(self):
        # Get face landmarks from current frame
        current_encoding = self.process_frame(draw=False)
        if not current_encoding:
            messagebox.showwarning("Warning", "No face detected. Try again.")
            return

        users = self.load_users()
        if not users:
            messagebox.showwarning("Warning", "No registered users found.")
            return

        current_normalized = self.normalize_and_align(current_encoding[0])

        best_match = (None, float('inf'))
        # Compare current face encoding with saved encodings
        for user, saved_encoding in users.items():
            saved = np.array(saved_encoding)
            euclidean = np.linalg.norm(saved - current_normalized)
            cosine_sim = np.dot(saved, current_normalized) / (np.linalg.norm(saved) * np.linalg.norm(current_normalized))
            score = euclidean * 0.4 + (1 - cosine_sim) * 0.6

            if score < best_match[1]:
                best_match = (user, score)

        if best_match[1] < 0.6:
            messagebox.showinfo("Success", f"Welcome, {best_match[0]}!")
        else:
            messagebox.showerror("Error", "Face not recognized!")

    def normalize_and_align(self, landmarks):
        """Align and normalize landmarks based on pose and scale"""
        landmarks = np.array(landmarks).reshape(-1, 3)

        # Reference points: left eye, right eye, nose tip
        left_eye = landmarks[468]
        right_eye = landmarks[473]
        nose_tip = landmarks[1]

        # Calculate rotation angle based on eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx)) - 180

        # Rotation matrix around z-axis
        rotation = R.from_euler('z', angle, degrees=True).as_matrix()

        # Center landmarks at nose tip
        centered = landmarks - nose_tip

        # Rotate points to align eyes horizontally
        aligned = np.dot(centered, rotation.T)

        # Normalize scale
        norm = np.linalg.norm(aligned)
        normalized = aligned / norm

        return normalized.flatten()

    def process_frame(self, draw=True):
        # Capture a frame from video source and process face landmarks
        ret, frame = self.vid.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        encodings = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                encoding = [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]
                encodings.append(encoding)

                if draw:
                    self.draw_landmarks(frame, face_landmarks)

        return encodings if encodings else None

    def draw_landmarks(self, frame, face_landmarks):
        # Draw face mesh landmarks and contours on frame
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )

    def save_data(self, data_type, key, data):
        # Load existing data or create new structure
        if os.path.exists(FACE_DATA_FILE):
            with open(FACE_DATA_FILE, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = {"captures": {}, "users": {}}

        existing_data[data_type][key] = data

        with open(FACE_DATA_FILE, "w") as f:
            json.dump(existing_data, f, indent=4)

    def load_users(self):
        # Load user data safely
        if not os.path.exists(FACE_DATA_FILE):
            return {}
        try:
            with open(FACE_DATA_FILE, "r") as f:
                data = json.load(f)
            return data.get("users", {})
        except (json.JSONDecodeError, IOError):
            return {}

    def update(self):
        # Continuously capture frames and update the Tkinter canvas
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            self.photo = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def cleanup(self):
        # Reset mode and hide capture button
        self.current_mode = None
        self.username = None
        self.btn_snapshot.grid_forget()

    def __del__(self):
        # Release video capture on app close
        if self.vid.isOpened():
            self.vid.release()

if __name__ == '__main__':
    FaceApp(tk.Tk(), "Face Recognition System")
