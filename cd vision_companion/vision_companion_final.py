import cv2
import pytesseract
from ultralytics import YOLO
import pyttsx3
import time
import re
import platform

# --- 1. INITIALIZATION ---

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)

# Select a clearer voice (female if available)
voices = engine.getProperty('voices')
if len(voices) > 1:
    engine.setProperty('voice', voices[1].id)
else:
    engine.setProperty('voice', voices[0].id)

# Set correct Tesseract path for macOS
if platform.system() == "Darwin":
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# Load YOLO model
try:
    model = YOLO("yolov8n.pt")  # Make sure yolov8n.pt is in this folder
except Exception as e:
    print(f"❌ ERROR loading YOLO model: {e}")
    engine.say("Failed to load object detection model.")
    engine.runAndWait()
    exit()

# Initialize camera (macOS compatibility)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("❌ Cannot access camera. Check permissions.")
    engine.say("Cannot access camera.")
    engine.runAndWait()
    exit()

# --- 2. INTRO MESSAGE ---

print("\n🎯 VISION COMPANION - FINAL HYBRID VERSION")
print("----------------------------------------------------")
print("Press 'O' - Object Detection Mode (Auto-speak)")
print("Press 'R' - Read Text (OCR Mode - Reads on R press)")
print("Press 'C' - Currency Detection Mode (Auto-detect)")
print("Press 'Q' - Quit")
print("----------------------------------------------------")

engine.say("Vision Companion started. Press O for objects, R for reading text, or C for currency detection.")
engine.runAndWait()

mode = None
last_spoken = ""
last_time = time.time()
SPEAK_DELAY = 5
last_currency = ""

# --- 3. HELPER FUNCTIONS ---

def speak(text):
    """Speaks the given text using pyttsx3"""
    engine.say(text)
    engine.runAndWait()

def detect_currency_value(text):
    """Detects common currency values in OCR text (supports rupees, dollars, euros)"""
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

# --- 4. MAIN LOOP ---

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    key = cv2.waitKey(1) & 0xFF

    # --- MODE SWITCHING ---

    if key == ord('o'):
        if mode != "object":
            mode = "object"
            print("🔍 Object Detection Mode Activated")
            speak("Object detection mode activated")

    elif key == ord('r'):
        if mode != "ocr":
            mode = "ocr"
            print("📄 Text Reading Mode Activated (Press R again to read text)")
            speak("Text reading mode activated. Press R again to read the visible text.")
        else:
            # Pressing R again triggers reading
            print("🔍 Reading text from current frame...")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)

            if text.strip():
                print("📄 Text Detected:\n", text)
                speak("The text says: " + text)
            else:
                print("⚠️ No readable text found.")
                speak("No text detected.")

    elif key == ord('c'):
        if mode != "currency":
            mode = "currency"
            print("💵 Currency Detection Mode Activated (Auto-detect)")
            speak("Currency detection mode activated.")
        else:
            mode = None
            print("🟡 Returning to idle mode")
            speak("Returning to idle mode")

    elif key == ord('q'):
        print("👋 Exiting Vision Companion...")
        speak("Goodbye")
        break

    # --- MODE-SPECIFIC LOGIC ---

    display_frame = frame

    if mode == "object":
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        display_frame = annotated_frame

        detected_names = set()
        for box in results[0].boxes:
            cls = int(box.cls[0])
            detected_names.add(results[0].names[cls])

        if detected_names and (time.time() - last_time > SPEAK_DELAY):
            spoken = ", ".join(detected_names)
            if spoken != last_spoken:
                print(f"👁️ Detected: {spoken}")
                speak(f"I see {spoken}")
                last_spoken = spoken
                last_time = time.time()

    elif mode == "ocr":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = gray
        cv2.putText(display_frame, "OCR Mode: Press 'R' again to read text.", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    elif mode == "currency":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = gray
        text = pytesseract.image_to_string(gray)
        detected_value = detect_currency_value(text)

        if detected_value and detected_value != last_currency:
            print(f"💰 Detected: {detected_value}")
            speak(f"This is a {detected_value}")
            last_currency = detected_value
            last_time = time.time()

        cv2.putText(display_frame, "Currency Mode: Show a note to detect value.", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    else:
        cv2.putText(display_frame, "Press 'O', 'R', or 'C' to start.", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "CLICK THIS WINDOW FIRST!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # --- DISPLAY FRAME ---
    cv2.imshow("💡 Vision Companion Dashboard", display_frame)

# --- 5. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
