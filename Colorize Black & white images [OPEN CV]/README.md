<div align="center">

# 🎨 Colorize Black & White Images — OpenCV Deep Learning

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-DNN-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Caffe](https://img.shields.io/badge/Caffe-Pre--trained%20Model-red?style=for-the-badge)](http://caffe.berkeleyvision.org/)
[![Tkinter](https://img.shields.io/badge/Tkinter-GUI%20App-blue?style=for-the-badge)](https://docs.python.org/3/library/tkinter.html)
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](../LICENSE.md)

> Automatically colorizes **black & white images** using a pre-trained deep learning model loaded via **OpenCV DNN** — wrapped in a clean **Tkinter desktop GUI** where you upload an image and get a colorized result instantly.

[🔙 Back to Main Repository](https://github.com/shsarv/Machine-Learning-Projects)

</div>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [How It Works](#-how-it-works)
- [The Science — Lab Color Space](#-the-science--lab-color-space)
- [The Model — Zhang et al. 2016](#-the-model--zhang-et-al-2016)
- [Model Files](#-model-files)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [App Preview](#-app-preview)
- [Tech Stack](#-tech-stack)
- [References & Citation](#-references--citation)

---

## 🔬 About the Project

Manually colorizing historical black & white photographs is an extremely time-consuming artistic process. This project automates it entirely using a **Convolutional Neural Network** trained to "hallucinate" plausible colors for any grayscale input — from old family photos to historical images.

Rather than training a model from scratch, the project loads **Richard Zhang et al.'s 2016 pre-trained Caffe model** directly through **OpenCV's DNN module**, making inference fast and dependency-light. The entire experience is wrapped in a **Tkinter GUI** where users upload a grayscale image and receive a colorized version with a single click.

**What this project covers:**
- Understanding Lab color space and why it is ideal for colorization
- Loading and running a pre-trained Caffe model via OpenCV DNN
- Image preprocessing: RGB → Lab, extracting the L channel as input
- Post-processing: merging predicted `ab` channels back with `L`, converting Lab → BGR
- Building a desktop GUI with Tkinter for real-time image upload and display

---

## ⚙️ How It Works

```
Input: Grayscale / B&W Image
              │
              ▼
   Convert: BGR → RGB → Lab
              │
              ▼
   Extract L channel (lightness only)
   Resize to 224 × 224
              │
              ▼
   OpenCV DNN Forward Pass
   (Zhang et al. Caffe model)
              │
              ▼
   Predict ab channels
   (313 quantized color bins → soft-decoded to ab)
              │
              ▼
   Resize predicted ab → original image size
              │
              ▼
   Concatenate: L (original) + ab (predicted)
              │
              ▼
   Convert: Lab → BGR
   Clip values to [0, 1], scale to [0, 255]
              │
              ▼
   Output: Colorized Image → Display in GUI / Save
```

---

## 🎨 The Science — Lab Color Space

This project uses the **Lab color space** rather than the familiar RGB. Here's why it matters:

| Channel | Represents | Role in This Project |
|---------|-----------|---------------------|
| **L** | Lightness (0 = black, 100 = white) | Input to the model — this IS the grayscale image |
| **a** | Green ↔ Red axis | Predicted by the neural network |
| **b** | Blue ↔ Yellow axis | Predicted by the neural network |

**The key insight:** In Lab, grayscale information is *entirely* encoded in the `L` channel. Color information lives only in `a` and `b`. This means the model only needs to learn to predict two channels from one — a much cleaner problem than mapping RGB to RGB.

```
Grayscale Image = L channel
                  │
         ┌────────┴────────┐
         ▼                 ▼
  Neural Network      (kept as-is)
  predicts: a, b          L
         │                 │
         └────────┬────────┘
                  ▼
           Lab image → BGR
           = Colorized Output
```

---

## 🧠 The Model — Zhang et al. 2016

The colorization model is from the landmark 2016 ECCV paper **"Colorful Image Colorization"** by Richard Zhang, Phillip Isola, and Alexei A. Efros (UC Berkeley).

**Key design choices in the paper:**

| Aspect | Detail |
|--------|--------|
| **Training data** | 1.3M images from ImageNet (Lab converted) |
| **Input** | L channel (grayscale), resized to 224×224 |
| **Output** | Predicted `ab` channels over 313 quantized color bins |
| **Loss function** | Multinomial cross-entropy with rebalanced class weights (to prevent desaturated outputs) |
| **Architecture** | Deep CNN with 8 conv blocks, no pooling — uses dilated convolutions to preserve spatial resolution |
| **Color decoding** | Annealed-mean of the 313 bin distribution (avoids washed-out grays from using the mean) |

> **Why 313 bins?** The `ab` color space is quantized into 313 bins with a grid size of 10. The model predicts a probability distribution over all 313 possible colors for each pixel, then decodes to a single `ab` value.

---

## 📦 Model Files

Three files are required to run inference. They are **not included** in the repository due to size and must be downloaded separately:

| File | Description | Download |
|------|-------------|---------|
| `colorization_release_v2.caffemodel` | Pre-trained model weights (~125 MB) | [Berkeley EECS](http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel) |
| `colorization_deploy_v2.prototxt` | Network architecture definition | [richzhang/colorization](https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt) |
| `pts_in_hull.npy` | 313 cluster center points in ab space | [richzhang/colorization](https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy?raw=true) |

Download all three with:

```bash
mkdir -p models

# Caffe model weights (~125 MB)
wget http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel \
     -O ./models/colorization_release_v2.caffemodel

# Network prototxt definition
wget https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt \
     -O ./models/colorization_deploy_v2.prototxt

# Cluster centers (ab quantization bins)
wget https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy?raw=true \
     -O ./pts_in_hull.npy
```

---

## 📁 Project Structure

```
Colorize Black & white images [OPEN CV]/
│
├── 📂 models/
│   ├── colorization_release_v2.caffemodel    # Pre-trained weights (download separately)
│   └── colorization_deploy_v2.prototxt       # Network architecture
│
├── pts_in_hull.npy                           # 313 ab color bin cluster centers
├── colorize.py                               # Core colorization logic (OpenCV DNN pipeline)
├── gui.py                                    # Tkinter GUI application
├── new.jpg                                   # Sample test image
├── result.png                                # Sample colorized output
├── requirements.txt                          # Python dependencies
└── README.md                                 # You are here
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shsarv/Machine-Learning-Projects.git
cd "Machine-Learning-Projects/Colorize Black & white images [OPEN CV]"
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Download model files

Run the wget commands from the [Model Files](#-model-files) section above, or download manually and place them in `./models/`.

### 4. Run the GUI app

```bash
python gui.py
```

This opens the Tkinter desktop window:
- **File → Upload Image** — select any grayscale or black & white `.jpg` / `.png`
- **File → Color Image** — run the colorization model and display the result

### 5. Run colorization directly (no GUI)

```bash
python colorize.py --image new.jpg
# Outputs: result.png in the current directory
```

---

## 🖥️ App Preview

```
┌──────────────────────────────────────────────────┐
│           B&W Image Colorization                 │
│  File ▾                                          │
│  ├── Upload Image                                │
│  └── Color Image                                 │
│                                                  │
│  ┌───────────────────┐  ┌───────────────────┐    │
│  │                   │  │                   │    │
│  │   [B&W Input]     │  │  [Colorized Out]  │    │
│  │                   │  │                   │    │
│  └───────────────────┘  └───────────────────┘    │
└──────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.7+ |
| Computer Vision | OpenCV (`cv2.dnn`) |
| Pre-trained Model | Caffe (Zhang et al. 2016) |
| GUI Framework | Tkinter |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |

---

## 📚 References & Citation

**Paper behind the model:**

```bibtex
@inproceedings{zhang2016colorful,
  title     = {Colorful Image Colorization},
  author    = {Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle = {ECCV},
  year      = {2016}
}
```

- [Colorful Image Colorization — Zhang et al. (2016)](https://arxiv.org/abs/1603.08511)
- [Official Demo & Model — richzhang/colorization](https://github.com/richzhang/colorization)
- [OpenCV DNN colorization sample](https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py)
- [PyImageSearch Tutorial — Adrian Rosebrock](https://pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/)

---

<div align="center">

Part of the [Machine Learning Projects](https://github.com/shsarv/Machine-Learning-Projects) collection by [Sarvesh Kumar Sharma](https://github.com/shsarv)

⭐ Star the main repo if this helped you!

</div>
