import mysql.connector
import cv2
import face_recognition
import base64

DATABASE_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'user_auth'
}

def insert_face_data(username, password):
    # Access webcam to capture face
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    video_capture.release()
    if not ret:
        print("Error accessing webcam.")
        return

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame)

    if not face_encodings:
        print("No face detected. Please try again.")
        return

    # Get the first face encoding
    face_encoding = face_encodings[0]
    encoded_face = base64.b64encode(face_encoding.tobytes()).decode()

    # Insert into MySQL database
    conn = mysql.connector.connect(**DATABASE_CONFIG)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (username, password, face_data) VALUES (%s, %s, %s)",
        (username, password, encoded_face)
    )
    conn.commit()
    conn.close()

    print(f"Face data for {username} inserted successfully.")

if __name__ == "__main__":
    username = input("Enter username: ")
    password = input("Enter password: ")
    insert_face_data(username, password)
