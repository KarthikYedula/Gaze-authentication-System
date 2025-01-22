import threading
import time
from flask import Flask, request, render_template, redirect, url_for, jsonify
import mysql.connector
import cv2
import face_recognition
import base64
import numpy as np
from gaze_tracking_module import authenticate_gaze, compare_coordinates  # Updated module
import os


is_tracking = False  # Tracks whether gaze tracking is currently active
terminate_gaze = False
maze_coords = []  # Global variable for maze coordinates
gaze_coords = []  # Global variable for gaze coordinates

app = Flask(__name__)

DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'user_auth'
}

def fetch_user(username, password):
    conn = mysql.connector.connect(**DATABASE_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user

def compare_faces(stored_face_data, detected_face_encoding):
    stored_encoding = np.frombuffer(base64.b64decode(stored_face_data), dtype=np.float64)
    return face_recognition.compare_faces([stored_encoding], detected_face_encoding)[0]

@app.route('/', methods=['GET', 'POST'])
def login():
    global is_tracking, terminate_gaze, maze_coords, gaze_coords

    # Reset state variables to allow a fresh start for gaze tracking
    is_tracking = False
    terminate_gaze = False
    maze_coords = []
    gaze_coords = []

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = fetch_user(username, password)
        if user:
            # Activate webcam for face detection
            video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            ret, frame = video_capture.read()
            video_capture.release()
            if not ret:
                return "Error accessing webcam. Please ensure the webcam is working."

            # Detect face in the frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame)

            if not face_encodings:
                return "No face detected. Please ensure your face is visible to the camera."

            detected_face_encoding = face_encodings[0]
            if compare_faces(user['face_data'], detected_face_encoding):
                # Redirect to Gaze Authentication
                return redirect(url_for('gaze_auth'))
            else:
                return "Face does not match. Access denied."
        else:
            return "Invalid username or password."

    return render_template('login.html')


@app.route('/gaze-auth', methods=['GET', 'POST'])
def gaze_auth():
    global terminate_gaze, is_tracking, maze_coords, gaze_coords

    if request.method == 'GET':
        # Start gaze tracking if not already running
        if not is_tracking:
            threading.Thread(target=authenticate_gaze, args=(lambda: terminate_gaze, maze_coords, gaze_coords)).start()
            is_tracking = True
        return render_template('gaze_auth.html')

    elif request.method == 'POST':  # Finish button clicked
        terminate_gaze = True  # Signal to stop gaze tracking

        if compare_coordinates(maze_coords, gaze_coords):
            return redirect(url_for('success'))
        else:
            return redirect(url_for('login'))  # Redirect to login page on failure



@app.route('/success')
def success():
    print("Authentication successful! Redirecting to success page...")
    return render_template('success.html')




@app.route('/finish', methods=['POST'])
def finish():
    # Compare coordinates and determine authentication result
    if compare_coordinates():
        return redirect(url_for('success'))
    else:
        return "Gaze authentication failed. Access denied."

@app.route('/terminate', methods=['POST'])
def terminate():
    print("Terminating session after successful authentication...")
    os._exit(0)  # Terminate the Flask server



if __name__ == '__main__':
    app.run(debug=True)
