from ultralytics import YOLO
import cv2
import pyttsx3
import time

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # small version for faster real-time performance

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access camera.")
    exit()

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)

print("✅ Object detection started. Press 'Q' to quit.")

last_spoken = ""
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Extract detected object names
    names = results[0].names
    detected = set()
    for box in results[0].boxes:
        cls = int(box.cls[0])
        detected.add(names[cls])

    # Display results
    cv2.imshow("Vision Companion - Object Detection", annotated_frame)

    # Speak detected objects every 5 seconds (to avoid repetition)
    if detected and (time.time() - last_time > 5):
        text_to_speak = ", ".join(detected)
        if text_to_speak != last_spoken:
            print(f"👁️ Detected: {text_to_speak}")
            engine.say(f"I see {text_to_speak}")
            engine.runAndWait()
            last_spoken = text_to_speak
            last_time = time.time()

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
