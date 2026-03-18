<div align="center">

# 🚗 Distracted Driver Detection — ResNet50 from Scratch

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-State%20Farm%20%7C%20Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
[![Classes](https://img.shields.io/badge/10%20Behavior%20Classes-orange?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](../LICENSE.md)

> Classifies **10 distracted driving behaviors** from dashboard camera images using a **custom ResNet50 implementation built from scratch in Keras** — including manual `convolutional_block` and `identity_block` definitions, `glorot_uniform` initialization, and LOGO cross-validation strategy.

[🔙 Back to Main Repository](https://github.com/shsarv/Machine-Learning-Projects)

</div>

---

## ⚠️ Safety Context

> Distracted driving causes thousands of road fatalities annually. Automated in-vehicle behavior classification from dashboard cameras is an active area of road safety AI research.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Class Definitions](#-class-definitions)
- [Model Architecture](#-model-architecture)
- [Training Analysis & Challenges](#-training-analysis--challenges)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [References](#-references)

---

## 🔬 About the Project

This project tackles the **State Farm Distracted Driver Detection** Kaggle challenge — classifying driver images into 10 behavior classes. What makes it distinctive is that **ResNet50 is implemented completely from scratch** using the Keras functional API, manually defining every bottleneck block and skip connection rather than using `tf.keras.applications`.

The notebook also demonstrates handling real-world ML challenges: **high bias**, **high variance**, and the **LOGO (Leave-One-Group-Out) cross-validation** strategy needed because multiple images belong to the same driver — random splits would leak the same driver into both train and validation sets.

**What this project covers:**
- Manual `identity_block` and `convolutional_block` implementations in Keras
- `resnets_utils` helper module for block definitions
- Diagnosing and addressing underfitting (high bias) and overfitting (high variance)
- LOGO cross-validation to prevent driver-level data leakage

---

## ⚙️ How It Works

```
Dashboard Camera Image
         │
         ▼
  Load + Preprocess
  (Normalize pixel values / 255)
         │
         ▼
  ResNet50 Forward Pass
  (Custom Keras implementation)
  ┌─────────────────────────────────┐
  │ ZeroPadding2D (3,3)             │
  │ Conv2D(64,7×7,s=2) → BN → ReLU │
  │ MaxPool(3×3, s=2)               │
  │ Stage 2: ConvBlock + IdBlock×2  │
  │ Stage 3: ConvBlock + IdBlock×3  │
  │ Stage 4: ConvBlock + IdBlock×5  │
  │ Stage 5: ConvBlock + IdBlock×2  │
  │ AveragePooling2D(2×2)           │
  │ Flatten → Dense(10, softmax)    │
  └─────────────────────────────────┘
         │
         ▼
  10-Class Softmax Output → c0–c9
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | State Farm Distracted Driver Detection |
| **Source** | [Kaggle Competition](https://www.kaggle.com/c/state-farm-distracted-driver-detection) |
| **Training Images** | 22,424 |
| **Classes** | 10 driving behaviors |
| **Input Shape** | Resized to `64 × 64 × 3` for training |
| **Metadata** | `driver_imgs_list.csv` — subject ID, classname, filename |
| **Key Challenge** | Multiple images per driver → LOGO cross-validation required |

---

## 🚦 Class Definitions

| Code | Behavior |
|:----:|----------|
| **c0** | ✅ Safe Driving |
| **c1** | 📱 Texting — Right Hand |
| **c2** | 📞 Phone Call — Right Hand |
| **c3** | 📱 Texting — Left Hand |
| **c4** | 📞 Phone Call — Left Hand |
| **c5** | 🎵 Operating Radio |
| **c6** | 🥤 Drinking |
| **c7** | 🔙 Reaching Behind |
| **c8** | 💄 Hair / Makeup |
| **c9** | 💬 Talking to Passenger |

---

## 🏗️ Model Architecture

The notebook defines **ResNet50 from scratch** — no pretrained weights, no `tf.keras.applications`:

```python
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D,
    BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D)
from keras.models import Model
from keras.initializers import glorot_uniform
from resnets_utils import *

def ResNet50(input_shape=(64, 64, 3), classes=10, init=glorot_uniform(seed=0)):
    """
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL
    -> CONVBLOCK -> IDBLOCK*2
    -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5
    -> CONVBLOCK -> IDBLOCK*2
    -> AVGPOOL -> TOPLAYER
    """
```

**Block types:**

| Block | Shape Change | Used When |
|-------|-------------|-----------|
| **Identity Block** | Input = Output shape | Deepening without dimension change |
| **Convolutional Block** | Input ≠ Output shape | When stride changes or filter count increases |

**Stage filter configurations:**

| Stage | Filters | Blocks |
|-------|---------|--------|
| Stage 2 | [64, 64, 256] | ConvBlock + IdBlock × 2 |
| Stage 3 | [128, 128, 512] | ConvBlock + IdBlock × 3 |
| Stage 4 | [256, 256, 1024] | ConvBlock + IdBlock × 5 |
| Stage 5 | [512, 512, 2048] | ConvBlock + IdBlock × 2 |

**Training config:**

| Parameter | Value |
|-----------|-------|
| Initializer | `glorot_uniform(seed=0)` |
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Input Shape | `(64, 64, 3)` |
| Output | Dense(10, softmax) |

---

## 📉 Training Analysis & Challenges

The notebook provides honest, detailed bias-variance analysis across training runs — a key learning documented in the project:

### Epoch 2 Results
| Set | Accuracy |
|-----|:--------:|
| Train | ~26% |
| Dev | ~13% |

> High bias (underfitting) — model hasn't converged. High variance — large gap between train/dev.

### Epoch 5 Results
| Set | Accuracy |
|-----|:--------:|
| Train | **37.83%** |
| Dev | **25.79%** |

> Train accuracy improved but **underfitting persists** (~62% away from 100%). Variance increased dramatically (+80% gap between epochs 2→5). The notebook diagnoses this explicitly:

```
"We still have an underfitting problem (high bias, about 62.17% from 100%),
however, our variance has increased dramatically between 2 and 5 epochs by about 80%."
```

### Prescribed fixes documented in the notebook:

**To address High Bias (underfitting):**
- Increase epoch count
- Use a bigger/deeper network
- Try different optimizers or learning rate schedules

**To address High Variance (overfitting):**
- Apply L2 regularization
- Add dropout layers
- Use data augmentation
- Increase training data volume

### LOGO Cross-Validation Note

> Standard random train/val splits cause **data leakage** — the same driver's images appear in both sets, inflating dev accuracy. The notebook flags this and recommends **Leave-One-Group-Out (LOGO)** cross-validation, splitting by `subject` (driver ID) from `driver_imgs_list.csv`.

---

## 📁 Project Structure

```
Distracted Driver Detection/
│
├── 📂 dataset/
│   ├── train/                         # Training images, organized by class
│   │   ├── c0/  c1/  c2/  ...  c9/
│   └── test/                          # Unlabeled test images
│
├── driver_imgs_list.csv               # subject, classname, img columns
├── resnets_utils.py                   # identity_block + convolutional_block helpers
├── distracted_driver_detection.ipynb  # Main notebook
├── requirements.txt                   # Python dependencies
└── README.md                          # You are here
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shsarv/Machine-Learning-Projects.git
cd "Machine-Learning-Projects/Distracted Driver Detection"
```

### 2. Download the dataset from Kaggle

```bash
pip install kaggle
kaggle competitions download -c state-farm-distracted-driver-detection
unzip state-farm-distracted-driver-detection.zip -d dataset/
```

Or download manually from: [kaggle.com/c/state-farm-distracted-driver-detection/data](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)

### 3. Set up environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook distracted_driver_detection.ipynb
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.7+ |
| Deep Learning | TensorFlow / Keras |
| Model | ResNet50 (from scratch via Keras functional API) |
| Utilities | `resnets_utils.py` (custom block helpers) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib |
| Notebook | Jupyter / Google Colab |

---

## 📚 References

- [State Farm Distracted Driver Detection — Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection)
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition.* [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- [deeplearning.ai — ResNet50 from scratch (Coursera)](https://www.coursera.org/learn/convolutional-neural-networks)
- [Keras Functional API Documentation](https://keras.io/guides/functional_api/)

---

<div align="center">

Part of the [Machine Learning Projects](https://github.com/shsarv/Machine-Learning-Projects) collection by [Sarvesh Kumar Sharma](https://github.com/shsarv)

⭐ Star the main repo if this helped you!

</div>
