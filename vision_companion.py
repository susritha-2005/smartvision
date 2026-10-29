import cv2
import pytesseract
from ultralytics import YOLO
import pyttsx3
import time
import re

# --- 1. INITIALIZATION ---

engine = pyttsx3.init()
engine.setProperty('rate', 170)

# Load YOLO model
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"❌ ERROR loading YOLO model: {e}. Check if yolov8n.pt is in the script directory.")
    exit()

# Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("❌ Cannot access camera. Check permissions.")
    engine.say("Cannot access camera.")
    engine.runAndWait()
    exit()

# --- 2. CONFIGURATION ---
print("\n🎯 VISION COMPANION STARTED")
print("----------------------------------------------------")
print("Press 'O' - Object Detection Mode (Auto-speak)")
print("Press 'R' - Read Text (OCR Mode - Reads on R press)")
print("Press 'C' - Currency Detection Mode")
print("Press 'Q' - Quit")
print("----------------------------------------------------")
engine.say("Vision companion started. Press O for objects, R for text, or C for currency detection.")
engine.runAndWait()

mode = None
last_spoken = ""
last_time = time.time()
SPEAK_DELAY = 5

# --- 3. CURRENCY DETECTION HELPER ---

def detect_currency_value(text):
    """
    Attempts to extract the value of a currency note from text.
    Supports INR, USD, EUR, etc.
    """
    text = text.lower()

    # Common patterns for different currencies
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
            engine.say("Object detection mode activated")
            engine.runAndWait()

    elif key == ord('r'):
        if mode != "ocr":
            mode = "ocr"
            print("📄 Text Reading Mode Activated")
            engine.say("Text reading view activated. Press R again to read the text.")
            engine.runAndWait()
        else:
            print("🔍 Reading text from frame...")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            if text.strip():
                print("📄 Text Detected:\n", text)
                engine.say("The text says: " + text)
                engine.runAndWait()
            else:
                print("⚠️ No readable text found.")
                engine.say("No text detected.")
                engine.runAndWait()

    elif key == ord('c'):
        if mode != "currency":
            mode = "currency"
            print("💵 Currency Detection Mode Activated")
            engine.say("Currency detection mode activated.")
            engine.runAndWait()
        else:
            print("🔎 Analyzing currency note...")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            detected_value = detect_currency_value(text)
            if detected_value:
                print(f"💰 Detected: {detected_value}")
                engine.say(f"This is a {detected_value}")
                engine.runAndWait()
            else:
                print("⚠️ No currency value detected.")
                engine.say("No currency value detected.")
                engine.runAndWait()

    elif key == ord('q'):
        print("👋 Exiting Vision Companion...")
        engine.say("Goodbye")
        engine.runAndWait()
        break

    # --- MODE-SPECIFIC DISPLAY ---
    display_frame = frame

    if mode == "object":
        results = model(frame, verbose=False)
        annotated = results[0].plot()
        display_frame = annotated

        detected_names = set()
        for box in results[0].boxes:
            cls = int(box.cls[0])
            detected_names.add(results[0].names[cls])

        if detected_names and (time.time() - last_time > SPEAK_DELAY):
            spoken = ", ".join(detected_names)
            if spoken != last_spoken:
                print(f"👁️ Detected: {spoken}")
                engine.say(f"I see {spoken}")
                engine.runAndWait()
                last_spoken = spoken
                last_time = time.time()

    elif mode == "ocr":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = gray
        cv2.putText(display_frame, "OCR Mode: Press 'R' to read text.", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    elif mode == "currency":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = gray
        cv2.putText(display_frame, "Currency Mode: Press 'C' to detect note value.", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    else:
        cv2.putText(display_frame, "Press 'O', 'R', or 'C' to start.", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "CLICK THIS WINDOW FIRST!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("💡 Vision Companion Dashboard", display_frame)

# --- 5. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
