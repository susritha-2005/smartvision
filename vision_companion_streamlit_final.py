import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import pyttsx3

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Vision Companion",
    layout="wide",
    page_icon="🦯"
)

st.title("🦯 Vision Companion")
st.write("Helping visually impaired users detect objects and read text in real-time.")

# -------------------------------
# Initialize Text-to-Speech
# -------------------------------
engine = pyttsx3.init()
voices = engine.getProperty('voices')

voice_options = {v.name: v.id for v in voices}
voice_choice = st.sidebar.selectbox("🎤 Choose Voice", options=list(voice_options.keys()))
engine.setProperty('voice', voice_options[voice_choice])
engine.setProperty('rate', 170)

# -------------------------------
# Mode Selection
# -------------------------------
mode = st.sidebar.radio(
    "Select Mode:",
    ["🧠 Object Detection", "📄 Text Reading (OCR)"]
)

st.sidebar.info("📷 Use your phone’s or laptop’s camera below.")

# -------------------------------
# Streamlit Camera Input
# -------------------------------
frame = st.camera_input("Take a picture or use your mobile camera")

# -------------------------------
# Helper Functions
# -------------------------------
def detect_objects(image):
    net = cv2.dnn.readNetFromCaffe(
        "MobileNetSSD_deploy.prototxt.txt",
        "MobileNetSSD_deploy.caffemodel"
    )

    class_names = [
        "background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"
    ]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    detected_items = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = class_names[idx]
            detected_items.append(label)

    return list(set(detected_items))


def read_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text.strip()

# -------------------------------
# Processing
# -------------------------------
if frame is not None:
    image = Image.open(frame)
    image = np.array(image)

    st.image(image, caption="Captured Frame", use_column_width=True)

    if mode == "🧠 Object Detection":
        st.subheader("Detected Objects:")
        detected_items = detect_objects(image)

        if detected_items:
            for item in detected_items:
                st.success(f"✅ {item}")
            sentence = f"I see {', '.join(detected_items)}."
            engine.say(sentence)
            engine.runAndWait()
        else:
            st.warning("No objects detected.")
            engine.say("No objects detected.")
            engine.runAndWait()

    elif mode == "📄 Text Reading (OCR)":
        st.subheader("Extracted Text:")
        text = read_text_from_image(image)

        if text:
            st.text_area("Detected Text", text, height=200)
            engine.say("The text says: " + text)
            engine.runAndWait()
        else:
            st.warning("No readable text found.")
            engine.say("No readable text found.")
            engine.runAndWait()

else:
    st.info("📷 Please capture an image using the camera above.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("💡 Developed by *Poojith & Team* — Vision Companion for the Visually Impaired.")
