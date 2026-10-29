import streamlit as st
import cv2
from ultralytics import YOLO
import pytesseract
import pyttsx3
import time
import re
import numpy as np
from googletrans import Translator

# --- PAGE CONFIG ---
st.set_page_config(page_title="Vision Companion", page_icon="🦯", layout="wide")
st.title("🦯 Vision Companion – Multilingual Assistive AI")
st.markdown("""
This app detects **objects**, reads **text**, and identifies **currency notes** —  
now with **multilingual voice support**.
""")

# --- Initialize Translator and TTS ---
translator = Translator()
engine = pyttsx3.init()
engine.setProperty('rate', 170)
voices = engine.getProperty('voices')
voice_names = [v.name for v in voices]

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")

chosen_voice = st.sidebar.selectbox("🎤 Select Voice", options=voice_names, index=0)
for v in voices:
    if v.name == chosen_voice:
        engine.setProperty('voice', v.id)
        break

# 🌍 Language Options
languages = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "French": "fr",
    "Spanish": "es"
}
chosen_lang = st.sidebar.selectbox("🌐 Select Language", list(languages.keys()), index=0)

mode = st.sidebar.selectbox("Choose Mode", ["Idle", "Object Detection", "Text Reading (OCR)", "Currency Detection"])
enable_voice = st.sidebar.checkbox("Enable Voice Output", value=True)
run = st.sidebar.toggle("▶️ Start Vision Companion")

# --- Load YOLO model ---
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

model = load_yolo()

# --- Helper Functions ---
def translate_text(text):
    if chosen_lang != "English":
        try:
            return translator.translate(text, dest=languages[chosen_lang]).text
        except:
            return text  # fallback if translation fails
    return text

def speak(text):
    if enable_voice:
        translated = translate_text(text)
        engine.say(translated)
        engine.runAndWait()

def detect_currency_value(text):
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

# --- Camera Stream ---
FRAME_WINDOW = st.image([])
status_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)  # Default Mac camera
    if not cap.isOpened():
        st.error("❌ Cannot access camera.")
        speak("Cannot access camera.")
        st.stop()

    st.success(f"✅ {mode} mode activated")

    last_spoken = ""
    last_currency = ""
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

        # --- Mode Handling ---
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

            output_text = "Detected Objects: " + ", ".join(detected) if detected else "No objects detected"

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

            if detected_value and detected_value != last_currency:
                speak(f"This is a {detected_value}")
                last_currency = detected_value
                last_time = time.time()
                output_text = f"Detected Currency: {detected_value}"
            else:
                output_text = "No currency detected"

        else:
            output_text = "Idle mode — choose a mode to begin"

        FRAME_WINDOW.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        status_placeholder.markdown(f"**🟢 Status:** {output_text}")

    cap.release()
    st.warning("🛑 Camera stopped.")
else:
    st.info("👆 Turn on 'Start Vision Companion' in the sidebar to begin.")
