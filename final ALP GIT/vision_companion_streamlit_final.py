import streamlit as st
import cv2
from ultralytics import YOLO
import pytesseract
import pyttsx3
import time
import re
import numpy as np
from PIL import Image
import platform
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Vision Companion", page_icon="🧠", layout="wide")

st.title("🧠 Vision Companion - Streamlit + Voice Control")
st.markdown("""
This app assists visually impaired users by detecting **objects**, **reading text**, and identifying **currency notes**.  
You can use **webcam** or **upload an image**, and even customize the **voice** and **speed** of speech!
""")

# --- INITIALIZE TTS ENGINE ---
engine = pyttsx3.init()
voices = engine.getProperty('voices')

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Controls")

# Mode & Input
mode = st.sidebar.selectbox("Choose Mode", ["Idle", "Object Detection", "Text Reading (OCR)", "Currency Detection"])
input_type = st.sidebar.radio("Input Type", ["Live Camera", "Upload Image"])

# Voice Customization
st.sidebar.subheader("🎙️ Voice Settings")

voice_names = []
for i, v in enumerate(voices):
    name = v.name if v.name else f"Voice {i}"
    voice_names.append(name)

selected_voice = st.sidebar.selectbox("Choose Voice", voice_names)
voice_rate = st.sidebar.slider("Speech Speed (Rate)", 100, 250, 170)
enable_voice = st.sidebar.checkbox("Enable Voice Output", value=True)

# Apply chosen voice
for v in voices:
    if v.name == selected_voice:
        engine.setProperty('voice', v.id)
        break
engine.setProperty('rate', voice_rate)

run = st.sidebar.toggle("▶️ Start Vision Companion")

# --- SETUP TESSERACT PATH ---
if platform.system() == "Darwin":
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
elif platform.system() == "Windows":
    _win_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_win_tesseract):
        pytesseract.pytesseract.tesseract_cmd = _win_tesseract

# --- LOAD YOLO MODEL ---
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

# --- HELPER FUNCTIONS ---
def speak(text):
    """Speaks text using the selected pyttsx3 voice and rate."""
    if enable_voice:
        engine.say(text)
        engine.runAndWait()

def detect_currency_value(text):
    """Detect currency denomination from OCR text."""
    text = text.lower()
    patterns = {
        "rupee": r"(\b10\b|\b20\b|\b50\b|\b100\b|\b200\b|\b500\b|\b2000\b)",
        "dollar": r"(\b1\b|\b5\b|\b10\b|\b20\b|\b50\b|\b100\b)",
        "euro": r"(\b5\b|\b10\b|\b20\b|\b50\b|\b100\b|\b200\b|\b500\b)"
    }
    for currency, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            return f"{match.group(1)} {currency} note"
    return None

# --- MAIN DASHBOARD SETUP ---
FRAME_WINDOW = st.image([])
status_placeholder = st.empty()
uploaded_image = None

if input_type == "Upload Image":
    uploaded_image = st.file_uploader("📤 Upload an image file", type=["jpg", "jpeg", "png"])

if run:
    # ---- IMAGE UPLOAD MODE ----
    if input_type == "Upload Image" and uploaded_image is not None:
        image = Image.open(uploaded_image)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        display_frame = frame.copy()
        output_text = ""

        if mode == "Object Detection":
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()
            display_frame = annotated_frame
            detected = set()
            for box in results[0].boxes:
                cls = int(box.cls[0])
                detected.add(results[0].names[cls])
            output_text = "Detected: " + ", ".join(detected) if detected else "No objects detected"
            if detected:
                speak("I see " + ", ".join(detected))

        elif mode == "Text Reading (OCR)":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            output_text = text.strip() if text.strip() else "No text detected"
            if text.strip():
                speak("The text says: " + text)
            display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        elif mode == "Currency Detection":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            detected_value = detect_currency_value(text)
            display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if detected_value:
                output_text = f"Detected: {detected_value}"
                speak(f"This is a {detected_value}")
            else:
                output_text = "No currency detected"

        FRAME_WINDOW.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        status_placeholder.markdown(f"**🟢 Status:** {output_text}")

    # ---- LIVE CAMERA MODE ----
    elif input_type == "Live Camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access camera.")
            speak("Cannot access camera.")
            st.stop()

        st.success(f"✅ {mode} mode activated")

        last_spoken = ""
        last_time = time.time()
        SPEAK_DELAY = 5

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            output_text = ""

            if mode == "Object Detection":
                results = model(frame, verbose=False)
                annotated_frame = results[0].plot()
                display_frame = annotated_frame
                detected = set()
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    detected.add(results[0].names[cls])
                if detected and (time.time() - last_time > SPEAK_DELAY):
                    text_to_speak = ", ".join(detected)
                    if text_to_speak != last_spoken:
                        speak(f"I see {text_to_speak}")
                        last_spoken = text_to_speak
                        last_time = time.time()
                output_text = "Detected: " + ", ".join(detected) if detected else "No objects detected"

            elif mode == "Text Reading (OCR)":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray)
                output_text = text.strip() if text.strip() else "No text detected"
                if text.strip() and (time.time() - last_time > SPEAK_DELAY):
                    speak("The text says: " + text)
                    last_time = time.time()
                display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            elif mode == "Currency Detection":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray)
                detected_value = detect_currency_value(text)
                display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if detected_value:
                    output_text = f"Detected: {detected_value}"
                    speak(f"This is a {detected_value}")
                else:
                    output_text = "No currency detected"

            else:
                output_text = "Idle mode — choose a mode to begin."

            FRAME_WINDOW.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            status_placeholder.markdown(f"**🟢 Status:** {output_text}")

        cap.release()
        st.warning("🛑 Camera stopped.")

else:
    st.info("👆 Turn on 'Start Vision Companion' in the sidebar to begin.")
