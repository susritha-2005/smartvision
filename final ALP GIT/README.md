# 👁️ Vision Companion – An Assistive AI for the Visually Impaired

### 🚀 Developed by Team Visionaries
**Team Members:**
- 🧑‍💻 Poojith Kusampudi  
- 🧑‍💻 [Teammate 2 Name]  
- 🧑‍💻 [Teammate 3 Name]  
- 🧑‍💻 [Teammate 4 Name]  

---

## 📖 Project Overview

**Vision Companion** is an assistive computer-vision application designed to help visually impaired individuals perceive their surroundings through **AI-driven object detection, text reading, and speech feedback**.

The app identifies objects in real time, reads printed text (such as signs, books, or currency notes), and speaks out the recognized information to the user — making everyday navigation and reading easier.

---

## 🧠 Key Features

- 🧩 **Object Detection Mode** – Detects and names objects in front of the camera using YOLOv8.  
- 📰 **Text Reading Mode** – Uses OCR (Tesseract) to extract text from the frame and reads it aloud.  
- 💵 **Currency Detection** – Recognizes common currency denominations for financial assistance.  
- 🎤 **Speech Output** – Converts detection results to audio using Text-to-Speech.  
- 🎛️ **Interactive Dashboard** – Streamlit-based web interface with voice settings and mode selection.  

---

## 🧰 Technologies Used

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python 3 |
| **Framework** | Streamlit |
| **Computer Vision** | OpenCV, YOLOv8 (Ultralytics) |
| **OCR Engine** | Tesseract |
| **Speech Engine** | pyttsx3 |
| **Deployment** | Streamlit Cloud |
| **Version Control** | GitHub |

---

## ⚙️ How It Works

1. **Object Mode (`O`)** – YOLOv8 detects objects in real-time from the webcam.  
2. **Text Mode (`R`)** – Captures the current frame, extracts text with Tesseract, and reads it aloud.  
3. **Currency Mode (`C`)** – Detects Indian currency notes and announces their denomination.  
4. **Speech Engine** – Uses `pyttsx3` for offline voice feedback.  
5. **Streamlit UI** – Provides a simple dashboard accessible through any browser.

---

## 🧪 Installation & Setup

### 1️⃣ Clone or Download the Repository
```bash
git clone https://github.com/<your-username>/vision-companion-streamlit.git
cd vision-companion-streamlit


