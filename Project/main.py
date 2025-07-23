import cv2
import pygame
import time

# Initialize alarm
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")  # Make sure alarm.mp3 is in your folder

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")  # More accurate than default eye

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access camera.")
    exit()

# Eye closed detection variables
eye_closed_frames = 0
eye_open_frames = 0
ALERT_THRESHOLD = 25  # Number of continuous closed frames to trigger alarm

font = cv2.FONT_HERSHEY_SIMPLEX

print("Press ESC to exit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect for natural viewing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes within face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)

        # Draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 255), 2)

        if len(eyes) >= 2:
            eye_open_frames += 1
            eye_closed_frames = 0
            cv2.putText(frame, "Eyes Open", (x, y - 10), font, 0.7, (0, 255, 0), 2)
        else:
            eye_closed_frames += 1
            eye_open_frames = 0
            cv2.putText(frame, "Eyes Closed", (x, y - 10), font, 0.7, (0, 0, 255), 2)

        # Draw rectangles around detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Trigger alarm after threshold
    if eye_closed_frames >= ALERT_THRESHOLD:
        cv2.putText(frame, "DROWSINESS ALERT!", (50, 50), font, 1.2, (0, 0, 255), 3)
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()
    else:
        pygame.mixer.music.stop()

    cv2.imshow("Drowsy Driver Detection", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()