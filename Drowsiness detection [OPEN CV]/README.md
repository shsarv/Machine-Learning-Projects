<div align="center">

# 😴 Driver Drowsiness Detection — OpenCV + Keras CNN

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Pygame](https://img.shields.io/badge/Pygame-Alarm-green?style=for-the-badge)](https://www.pygame.org/)
[![Real-Time](https://img.shields.io/badge/Real--Time-Webcam-brightgreen?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](../LICENSE.md)

> A **real-time driver drowsiness detection system** that uses **Haar Cascade classifiers** to locate the driver's eyes in every webcam frame and a **custom-trained CNN** (`cnnCat2.h5`) to classify each eye as **Open** or **Closed** — sounding a `pygame` alarm when drowsiness is detected.

[🔙 Back to Main Repository](https://github.com/shsarv/Machine-Learning-Projects)

</div>

---

## ⚠️ Safety Context

> Drowsy driving causes thousands of road fatalities annually. This system provides a real-time, automated alert to combat driver fatigue using a lightweight CNN that runs entirely on a standard webcam feed.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [How It Works](#-how-it-works)
- [CNN Model Architecture](#-cnn-model-architecture)
- [Dataset](#-dataset)
- [Haar Cascade Files](#-haar-cascade-files)
- [Scoring & Alert Logic](#-scoring--alert-logic)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [Known Limitations](#-known-limitations)
- [References](#-references)

---

## 🔬 About the Project

This project detects driver drowsiness through a two-stage pipeline:

1. **Detection** — OpenCV Haar Cascade classifiers locate the face and each eye (left, right) in every frame
2. **Classification** — A custom-trained Keras CNN (`cnnCat2.h5`) classifies each eye ROI as **Open** or **Closed**

A running score is incremented each frame when eyes are detected as closed. When the score crosses a threshold, `pygame` plays `alarm.wav` and a "**DROWSY**" warning is overlaid on the video feed.

**What this project covers:**
- Training a binary CNN classifier on a custom ~7,000-image eye dataset
- Real-time face and eye detection with OpenCV Haar cascades
- Score-based drowsiness logic (accumulate → threshold → alarm)
- Alarm playback with `pygame.mixer`

---

## ⚙️ How It Works

```
Webcam Frame (live stream)
         │
         ▼
  Convert BGR → Grayscale
         │
         ▼
  Haar Cascade: Detect Face
  (haarcascade_frontalface_alt.xml)
         │
         ▼
  Haar Cascade: Detect Eyes from frame
  ├── Left Eye  (haarcascade_lefteye_2splits.xml)
  └── Right Eye (haarcascade_righteye_2splits.xml)
         │
         ▼
  Crop Eye ROI → Resize → Normalize
         │
         ▼
  CNN Forward Pass (cnnCat2.h5)
  → Predict: ['Close', 'Open']
  → rpred / lpred updated per frame
         │
         ├── Both eyes Open  → score decremented (min 0)
         │
         └── Eye(s) Closed   → score incremented
                   │
                   └── score > threshold
                             │
                             ▼
                      🔔 pygame alarm.wav
                      📺 "DROWSY" on screen
                      🟥 Red border on frame
```

---

## 🧠 CNN Model Architecture

`model.py` defines and trains the CNN classifier. The trained weights are saved as `models/cnnCat2.h5`.

```
Input: Eye ROI image (24 × 24 × 1, grayscale)
         │
         ▼
Conv2D(32, 3×3) → ReLU → MaxPool(1,1)
Conv2D(32, 3×3) → ReLU → MaxPool(1,1)
Conv2D(64, 3×3) → ReLU → MaxPool(1,1)
         │
         ▼
Flatten
Dense(128) → ReLU
Dropout(0.5)
Dense(2) → Softmax
         │
         ▼
Output: ['Close', 'Open']
```

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Classes | 2 — `Close` / `Open` |
| Input Size | 24 × 24 × 1 (grayscale) |
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Activation (hidden) | ReLU |
| Activation (output) | Softmax |
| Regularization | Dropout (0.5) |

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Type** | Custom — captured via webcam script |
| **Total Images** | ~7,000 eye images |
| **Classes** | `Open` / `Close` |
| **Conditions** | Various lighting conditions |
| **Cleaning** | Manually cleaned to remove unusable frames |

The dataset was created by writing a capture script that crops eye regions frame by frame and saves them to disk, labeled by folder (`Open/` or `Closed/`). It was then manually reviewed to remove noisy or ambiguous images.

> **Want to train on your own data?** Run `model.py` against your own captured eye dataset following the same `Open/Close` folder structure.

---

## 📂 Haar Cascade Files

Three XML classifiers are used from the `haar cascade files/` folder:

| File | Purpose |
|------|---------|
| `haarcascade_frontalface_alt.xml` | Detects the driver's face bounding box |
| `haarcascade_lefteye_2splits.xml` | Detects the left eye region within the frame |
| `haarcascade_righteye_2splits.xml` | Detects the right eye region within the frame |

These are pre-trained OpenCV Haar cascades — no training required. They are loaded in `drowsinessdetection.py` as:

```python
face  = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye  = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye  = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')
```

---

## 🎯 Scoring & Alert Logic

The system uses a **running score counter** rather than a fixed-frame threshold:

```python
lbl = ['Close', 'Open']   # CNN output labels

# Per frame:
if rpred[0] == 0 and lpred[0] == 0:   # Both eyes closed
    score += 1
    cv2.putText(frame, "Closed", ...)
else:                                  # Eyes open
    score -= 1
    cv2.putText(frame, "Open", ...)

score = max(score, 0)                  # Score never goes negative

if score > 15:                         # Drowsiness threshold
    # Sound alarm
    mixer.Sound('alarm.wav').play()
    # Draw red border on frame
    thicc = min(thicc + 2, 16)
    cv2.rectangle(frame, (0,0), (width,height), (0,0,255), thicc)
```

| Variable | Value | Meaning |
|----------|:-----:|---------|
| `score` threshold | **15** | Frames of closed eyes before alarm |
| `rpred` / `lpred` | `0` = Closed, `1` = Open | CNN prediction per eye |
| Border thickness `thicc` | Grows up to 16px | Visual urgency indicator |

---

## 📁 Project Structure

```
Drowsiness detection [OPEN CV]/
│
├── 📂 haar cascade files/
│   ├── haarcascade_frontalface_alt.xml     # Face detector
│   ├── haarcascade_lefteye_2splits.xml     # Left eye detector
│   └── haarcascade_righteye_2splits.xml    # Right eye detector
│
├── 📂 models/
│   └── cnnCat2.h5                          # Trained CNN weights (download separately)
│
├── drowsinessdetection.py                  # Main script — webcam loop + detection + alarm
├── model.py                                # CNN model definition + training script
├── alarm.wav                               # Alert sound file
└── README.md                               # You are here
```

> **Note:** `models/cnnCat2.h5` is not included in the repo due to GitHub file size limits. Download it from the Google Drive link in the project or train your own by running `model.py`.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shsarv/Machine-Learning-Projects.git
cd "Machine-Learning-Projects/Drowsiness detection [OPEN CV]"
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Download the trained model

The `cnnCat2.h5` model file must be placed in the `models/` folder. Download it from the link provided in the repository issues/releases, then:

```bash
mkdir models
# Place cnnCat2.h5 inside models/
```

Or train your own model from scratch:

```bash
python model.py
# Saves models/cnnCat2.h5 automatically
```

### 4. Run the detector

```bash
python drowsinessdetection.py
```

- The webcam opens automatically
- Eyes detected as closed → score increments
- Score exceeds threshold → **alarm sounds + red border appears**
- Press **`q`** to quit

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.7+ |
| Computer Vision | OpenCV (`cv2`) |
| Eye Detection | Haar Cascade Classifiers |
| Deep Learning | Keras + TensorFlow backend |
| Model | Custom CNN (`cnnCat2.h5`) |
| Audio Alarm | Pygame (`pygame.mixer`) |
| Numerical Processing | NumPy |

---

## ⚠️ Known Limitations

| Limitation | Detail |
|-----------|--------|
| **Lighting sensitivity** | Haar cascades and CNN accuracy drop under poor or uneven lighting |
| **Glasses / sunglasses** | Frames and tinted lenses obstruct eye detection |
| **Head pose** | Extreme angles may cause Haar cascade face/eye detection to fail |
| **Single eye closure** | If only one eye closes (winking), score increments only partially |
| **No yawn detection** | Fatigue from yawning is not measured — only eye closure |

---

## 📚 References

- [OpenCV Haar Cascade Documentation](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [Keras Documentation](https://keras.io/)
- [Pygame mixer Documentation](https://www.pygame.org/docs/ref/mixer.html)

---

<div align="center">

Part of the [Machine Learning Projects](https://github.com/shsarv/Machine-Learning-Projects) collection by [Sarvesh Kumar Sharma](https://github.com/shsarv)

⭐ Star the main repo if this helped you!

</div>
