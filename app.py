import whisper
import pyaudio
import wave
import tempfile
import streamlit as st
import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer

# Load pre-trained Whisper model
model_whisper = whisper.load_model("base")

mp_drawing = mp.solutions.drawing_utils  # Face and body landmarks
mp_holistic = mp.solutions.holistic  # Mediapipe Solutions

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Set up Streamlit
st.title("Emotion, Pose, and Live Transcription Detection")
st.sidebar.title("Select Interviewee and Resume")
interviewees = ["Hoang", "Fidan", "Ojashwi"]
resume_paths = {
    "Hoang": "C:/Users/dang0/Downloads/hoangResume.pdf",
    "Fidan": "fidan.pdf",
    "Ojashwi": "ojashwi.pdf"
}
selected_interviewee = st.sidebar.selectbox("Choose an Interviewee:", interviewees)

# Declare PDF upload logic
if 'pdf_ref' not in ss:
    ss.pdf_ref = None

st.file_uploader("Upload PDF file", type=('pdf'), key='pdf')

if ss.pdf:
    ss.pdf_ref = ss.pdf

if ss.pdf_ref:
    binary_data = ss.pdf_ref.getvalue()
    pdf_viewer(input=binary_data, width=700)

# Create placeholders for webcam and transcription
FRAME_WINDOW = st.image([])
TRANSCRIPT_WINDOW = st.empty()

# Checkbox to toggle webcam
run = st.checkbox("Start Webcam", key="start_webcam_checkbox")

# Button to control live audio transcription
start_transcription = st.button("Start Audio Transcription")
stop_transcription = st.button("Stop Audio Transcription")

# Button to display final transcript
display_transcript = st.button("Display Final Transcript")

# Transcription storage
if "transcription_text" not in ss:
    ss.transcription_text = []

# Function to record audio and transcribe using Whisper
def record_and_transcribe():
    chunk = 1024  # Audio chunk size
    format = pyaudio.paInt16  # 16-bit audio format
    channels = 1  # Mono audio
    rate = 16000  # Sample rate
    record_seconds = 10  # Record duration
    audio = pyaudio.PyAudio()

    # Temporary file for audio storage
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    # Start recording
    stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    st.write("Recording audio...")
    frames = []

    # Record chunks
    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save audio to a temporary file
    with wave.open(temp_audio_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    # Transcribe the audio with Whisper
    st.write("Transcribing audio...")
    result = model_whisper.transcribe(temp_audio_file)
    os.remove(temp_audio_file)  # Clean up temporary file
    return result['text']

# Initialize webcam
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

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Update the webcam feed
        FRAME_WINDOW.image(image)

    cap.release()  # Release the webcam when done
else:
    st.write("Click 'Start Webcam' to display the feed")

# Start transcription
if start_transcription:
    text = record_and_transcribe()
    ss.transcription_text.append(text)
    TRANSCRIPT_WINDOW.text_area("Live Transcription:", value="\n".join(ss.transcription_text), height=300)

# Stop transcription
if stop_transcription:
    st.write("Audio transcription stopped.")

# Display final transcript after the interview
if display_transcript:
    st.write("### Final Transcript:")
    st.text_area("Interview Transcript:", value="\n".join(ss.transcription_text), height=500)
