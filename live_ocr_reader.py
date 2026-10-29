import cv2
import pytesseract
import pyttsx3

# Initialize text-to-speech
tts = pyttsx3.init()
tts.setProperty('rate', 170)  # adjust speaking speed if needed

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access camera. Check permissions in System Settings > Security & Privacy.")
    exit()

print("✅ Camera started. Press 'R' to read text, 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Display the camera feed
    cv2.imshow("Vision Companion OCR (Press R to read text)", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):  # 'r' key to read text
        print("🔍 Reading text...")
        # Convert frame to grayscale for better OCR accuracy
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)

        if text.strip():
            print("📄 Text Detected:\n", text)
            tts.say("The text says: " + text)
            tts.runAndWait()
        else:
            print("⚠️ No readable text found.")
            tts.say("No text detected.")
            tts.runAndWait()

    elif key == ord('q'):  # 'q' to quit
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
