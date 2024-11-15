import pickle
import os
import cv2
import streamlit as st
import mediapipe as mp
import pandas as pd
import numpy as np

mp_drawing = mp.solutions.drawing_utils # face and body landmarks
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Live Emotion Recognition")
st.write("Using your webcam, this app detects and predicts emotions in real-time.")
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([]) 

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Set up Streamlit
st.title("Emotion and Pose Detection with Webcam")

# Create an image placeholder to display the webcam feed
FRAME_WINDOW = st.image([])

# Create a container to display the predictions
predictions_container = st.empty()

# Store predictions in a list
predictions = []

if run:
    cap = cv2.VideoCapture(0)  # Open webcam

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break
        
        # Recolor the frame (BGR to RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detections
        results = holistic.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks for face, hands, and pose
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
        
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        # Extract and process the pose and face landmarks for classification
        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            
            row = pose_row + face_row
            
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            
            # Display the class and probability on the frame
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                            [640, 480]).astype(int))
            
            cv2.rectangle(image, (coords[0], coords[1] + 5), 
                          (coords[0] + len(body_language_class) * 20, coords[1] - 30), 
                          (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display status box for class and probability
            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
            cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Store prediction result in the list
            predictions.append({
                "Class": body_language_class,
                "Probability": round(body_language_prob[np.argmax(body_language_prob)], 2)
            })

        except:
            pass

        # Update the webcam feed in Streamlit
        FRAME_WINDOW.image(image)

        # Display the predictions in Streamlit as a table
        with predictions_container:
            st.write("### Predictions per Frame:")
            st.write(pd.DataFrame(predictions))

    cap.release()  # Release the webcam when done
else:
    st.write("Click 'Start Webcam' to display the feed")
