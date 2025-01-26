import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Directories
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"

# Initialize known face encodings and names
known_face_encodings = []
known_face_names = []

# Load known faces
print("Loading known faces...")
for file_name in os.listdir(KNOWN_FACES_DIR):
    if file_name.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(KNOWN_FACES_DIR, file_name)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)

        if encoding:  # Check if encoding was successful
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(file_name)[0])
            print(f"Loaded {file_name}.")
        else:
            print(f"Warning: No face detected in {file_name}.")

# Function to mark attendance
def mark_attendance(name):
    try:
        # Create the file if it doesn't exist
        if not os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, "w") as file:
                file.write("Name,Date,Time\n")

        # Read existing attendance data
        with open(ATTENDANCE_FILE, "r") as file:
            existing_data = file.readlines()

        today = datetime.now().strftime('%d-%m-%Y')
        if any(line.startswith(f"{name},{today}") for line in existing_data):
            return

        # Append attendance data
        now = datetime.now()
        time_str = now.strftime('%H:%M:%S')
        with open(ATTENDANCE_FILE, "a") as file:
            file.write(f"{name},{today},{time_str}\n")
        print(f"Attendance marked for {name}")
    except Exception as e:
        print(f"Error in marking attendance: {e}")

# Initialize webcam
print("Starting video capture...")
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Unable to access the camera.")
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Unable to read frame. Exiting.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    try:
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        print(f"Detected {len(face_locations)} face(s).")

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Match faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

            name = "Unknown"
            if best_match_index is not None and matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Mark attendance
            if name != "Unknown":
                mark_attendance(name)

            # Draw rectangle around the face
            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error during face recognition: {e}")

    # Display the video feed
    cv2.imshow("Face Recognition Attendance", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program.")
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()