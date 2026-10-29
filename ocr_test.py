from PIL import Image
import pytesseract
import pyttsx3

img = Image.open("sample_text.jpg")
text = pytesseract.image_to_string(img)

print("Extracted text:")
print(text)

tts = pyttsx3.init()
tts.say("The text says: " + text)
tts.runAndWait()
