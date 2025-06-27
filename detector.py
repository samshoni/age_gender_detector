import cv2
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

# === Model paths ===
AGE_MODEL = "models/age_gender/age_net.caffemodel"
AGE_PROTO = "models/age_gender/age_deploy.prototxt"
GENDER_MODEL = "models/age_gender/gender_net.caffemodel"
GENDER_PROTO = "models/age_gender/gender_deploy.prototxt"

# === Labels ===
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# === Load Age & Gender models ===
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

# === Load Emotion model using hsemotion ===
fer = HSEmotionRecognizer(model_name='enet_b2_8')  # Can also use enet_b0_8, enet_b1_8, etc.

# === Load Haar cascade ===
face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')

# === Start webcam ===
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()

        # --- Age & Gender Detection ---
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)
        gender_net.setInput(blob)
        gender = GENDER_LIST[gender_net.forward().argmax()]

        age_net.setInput(blob)
        age = AGE_BUCKETS[age_net.forward().argmax()]

        # --- Emotion Detection ---
        try:
            emotion, scores = fer.predict_emotions(face_img, logits=False)
        except Exception as e:
            emotion = "Unknown"

        # --- Draw outputs ---
        label = f"{gender}, {age}, {emotion}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Age, Gender & Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
        break

cap.release()
cv2.destroyAllWindows()

