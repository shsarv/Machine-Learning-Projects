<div align="center">

# 🧠 Brain Tumor Detection — End to End

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![ResNet50](https://img.shields.io/badge/ResNet50-Transfer%20Learning-blueviolet?style=for-the-badge)](https://pytorch.org/vision/stable/models/resnet.html)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.3%25-brightgreen?style=for-the-badge)](https://github.com/shsarv/Machine-Learning-Projects/tree/main/BRAIN%20TUMOR%20DETECTION%20%5BEND%202%20END%5D)
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](../LICENSE.md)

> A full **end-to-end deep learning web application** that classifies brain tumors in MRI scans into three tumor types using a **fine-tuned ResNet50** via Transfer Learning — achieving **99.3% accuracy** — deployed as a live Flask web app.

[🔙 Back to Main Repository](https://github.com/shsarv/Machine-Learning-Projects)

</div>

---

## ⚠️ Medical Disclaimer

> **This tool is for educational and research purposes only.** It is not a substitute for professional medical diagnosis. Always consult a qualified radiologist or medical professional for clinical decisions.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [App Preview](#-app-preview)
- [Tech Stack](#-tech-stack)
- [References & Citation](#-references--citation)

---

## 🔬 About the Project

Brain tumors are among the most critical conditions in medicine — accurate classification directly guides treatment decisions (surgery, radiation, chemotherapy). This project demonstrates how **Transfer Learning** with a pre-trained **ResNet50** can achieve near-perfect classification accuracy on MRI scans, far outperforming a CNN trained from scratch.

The model is trained on the **Jun Cheng Figshare brain tumor dataset** (3,064 T1-weighted CE-MRI images from 233 patients) and deployed as a **Flask web application** where users can upload an MRI image and receive a real-time classification with confidence score.

**What this project covers:**
- Converting `.mat` (MATLAB) MRI files to images and extracting tumor masks/borders
- Data augmentation with custom real-time transformations
- Fine-tuning ResNet50 with Transfer Learning using PyTorch
- Model evaluation with accuracy, loss curves, and per-class metrics
- Serving predictions via a Flask web app

---

## ⚙️ How It Works

```
User Uploads MRI Scan (.jpg / .png)
              │
              ▼
    Image Preprocessing
  (Resize 224×224 → Normalize → Tensor)
              │
              ▼
   Fine-tuned ResNet50 Forward Pass
  (ImageNet weights → custom classifier head)
              │
              ▼
    3-Class Softmax Output
  ┌───────────┬─────────────┬────────────┐
  │  Glioma   │ Meningioma  │ Pituitary  │
  └───────────┴─────────────┴────────────┘
              │
              ▼
  Predicted Class + Confidence Score
       Displayed in Browser
```

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | Brain Tumor Dataset |
| **Author** | Jun Cheng |
| **Source** | [Figshare — DOI: 10.6084/m9.figshare.1512427](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) |
| **Total Images** | 3,064 T1-weighted CE-MRI scans |
| **Patients** | 233 |
| **Format** | `.mat` (MATLAB) → converted to `.jpg` |
| **Task** | Multi-class classification (3 tumor types) |

### Class Distribution

| Class | Description | Slices |
|-------|-------------|:------:|
| 🔴 **Glioma** | Arises from glial cells; most common & aggressive brain tumor | 1,426 |
| 🟡 **Meningioma** | Grows on membranes surrounding the brain; often benign | 708 |
| 🟢 **Pituitary** | Forms on the pituitary gland at the brain's base; usually slow-growing | 930 |
| **Total** | | **3,064** |

### `.mat` File Structure

Each `.mat` file contains the following fields:

| Field | Description |
|-------|-------------|
| `cjdata.image` | The MRI scan image matrix |
| `cjdata.label` | Tumor type label (1=Meningioma, 2=Glioma, 3=Pituitary) |
| `cjdata.tumorBorder` | Coordinates of discrete points on the tumor border `[x1,y1,x2,y2,...]` |
| `cjdata.tumorMask` | Binary image with `1s` marking the tumor region |

### Data Augmentation

Custom real-time augmentations applied during training:

| Technique | Purpose |
|-----------|---------|
| Horizontal & Vertical Flip | Positional variance |
| Random Rotation (±15°) | Scan orientation variance |
| Brightness / Contrast Jitter | Scanner setting variance |
| Random Crop / Zoom | Variable tumor scale |
| Normalization (ImageNet μ/σ) | Stable gradient flow |

---

## 🏗️ Model Architecture

Rather than training a CNN from scratch, this project applies **Transfer Learning** by fine-tuning a **ResNet50** pretrained on ImageNet.

```
Input MRI Image (224 × 224 × 3)
          │
          ▼
┌──────────────────────────────────────┐
│        ResNet50 Backbone             │
│   (Pretrained on ImageNet)           │
│                                      │
│  Conv1 → BN → ReLU → MaxPool        │
│  Layer1: 3× Bottleneck blocks        │
│  Layer2: 4× Bottleneck blocks        │
│  Layer3: 6× Bottleneck blocks        │
│  Layer4: 3× Bottleneck blocks        │
│  Adaptive AvgPool → 2048-dim vector  │
└──────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────┐
│     Custom Classifier Head           │
│  FC (2048 → 512) + ReLU             │
│  Dropout (0.5)                       │
│  FC (512 → 3)                        │
│  Softmax                             │
└──────────────────────────────────────┘
          │
          ▼
  Glioma / Meningioma / Pituitary
```

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Base Model | ResNet50 (ImageNet pretrained) |
| Strategy | Fine-tune full network after warm-up |
| Optimizer | Adam |
| Learning Rate | 0.001 (backbone), 0.01 (head) |
| LR Scheduler | StepLR (decay every 7 epochs) |
| Loss Function | Cross-Entropy Loss |
| Epochs | 25 |
| Batch Size | 32 |
| Train / Val / Test Split | 70% / 15% / 15% |

---

## 📈 Model Performance

| Metric | Score |
|--------|:-----:|
| **Overall Accuracy** | **~99.3%** |
| **Glioma F1** | ~99% |
| **Meningioma F1** | ~98% |
| **Pituitary F1** | ~99% |

> **Why Transfer Learning?** ResNet50 pre-trained on ImageNet already understands low-level features (edges, textures, shapes) that transfer well to MRI images. Fine-tuning requires far less data and training time while achieving significantly higher accuracy than training from scratch.

> Meningioma shows slightly lower scores due to class imbalance (708 vs 1,426 glioma images) and its high visual similarity to surrounding tissue.

---

## 📁 Project Structure

```
BRAIN TUMOR DETECTION [END 2 END]/
│
├── 📂 Dataset/
│   ├── 📂 bt_images/                        # Converted MRI images (.jpg)
│   ├── 📂 bt_mask/                          # Tumor mask images
│   └── 📂 new_dataset/                      # Processed dataset with labels
│
├── 📂 Model/
│   └── brain_tumor_model.pt                 # Saved ResNet50 fine-tuned weights
│
├── 📂 notebooks/
│   ├── brain_tumor_dataset_preparation.ipynb  # .mat → .jpg conversion & preprocessing
│   └── torch_brain_tumor_classifier.ipynb     # Training, evaluation & results
│
├── 📂 static/
│   ├── 📂 css/style.css                     # App styling
│   └── 📂 uploads/                          # Temporarily stores uploaded MRI scans
│
├── 📂 templates/
│   ├── index.html                           # Upload page
│   └── result.html                          # Prediction result page
│
├── app.py                                   # Flask application entry point
├── test.py                                  # CLI script: classify a single image by path
├── requirements.txt                         # Python dependencies
└── README.md                                # You are here
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shsarv/Machine-Learning-Projects.git
cd "Machine-Learning-Projects/BRAIN TUMOR DETECTION [END 2 END]"
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Download & prepare the dataset

The raw dataset is in `.mat` (MATLAB) format, split across 4 zip files on Figshare:

```bash
# Download all 4 parts from Figshare
# https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

# After downloading, extract each zip:
unzip brainTumorDataPublic_1-766.zip    -d dataset/bt_set1
unzip brainTumorDataPublic_767-1532.zip -d dataset/bt_set2
unzip brainTumorDataPublic_1533-2298.zip -d dataset/bt_set3
unzip brainTumorDataPublic_2299-3064.zip -d dataset/bt_set4
```

Then run the **dataset preparation notebook** to convert `.mat` → `.jpg` and generate labels:

```bash
jupyter notebook notebooks/brain_tumor_dataset_preparation.ipynb
```

### 4. Train the model (optional — pretrained weights included)

```bash
jupyter notebook notebooks/torch_brain_tumor_classifier.ipynb
```

### 5. Run the Flask app

```bash
python app.py
```

Navigate to → **http://127.0.0.1:5000** and upload an MRI scan.

### 6. Quick CLI prediction

```bash
python test.py --image path/to/mri_scan.jpg
```

---

## 🖥️ App Preview

```
┌──────────────────────────────────────────────────┐
│           🧠 Brain Tumor Classifier               │
│                                                  │
│   Upload a T1-weighted CE-MRI scan:              │
│   ┌────────────────────────────────────┐         │
│   │  [ Choose File ]   mri_scan.jpg    │         │
│   └────────────────────────────────────┘         │
│                                                  │
│              [ Analyze Scan ]                    │
│                                                  │
│  ──────────────────────────────────────────────  │
│                                                  │
│   Result:  🔴  GLIOMA                            │
│   Confidence:  98.7%                             │
│                                                  │
│   ⚠️  Please consult a medical professional.     │
└──────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.7+ |
| Deep Learning | PyTorch, Torchvision |
| Model | ResNet50 (Transfer Learning) |
| Image Processing | OpenCV, PIL (Pillow) |
| Data / EDA | Pandas, NumPy, Matplotlib, Seaborn, h5py |
| Web Framework | Flask |
| Frontend | HTML5, CSS3, Bootstrap |
| Model Serialization | `torch.save` / `.pt` |
| Notebook | Jupyter / Google Colab |

---

## 📚 References & Citation

**Dataset — please cite if you use this work:**

```bibtex
@article{Cheng2015,
  author  = {Cheng, Jun and others},
  title   = {Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition},
  journal = {PLoS ONE},
  volume  = {10},
  number  = {10},
  year    = {2015}
}

@article{Cheng2016,
  author  = {Cheng, Jun and others},
  title   = {Retrieval of Brain Tumors by Adaptive Spatial Pooling and Fisher Vector Representation},
  journal = {PLoS ONE},
  volume  = {11},
  number  = {6},
  year    = {2016}
}
```

**Further reading:**
- [Jun Cheng Brain Tumor Dataset — Figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- [Deep Residual Learning for Image Recognition — He et al. (2015)](https://arxiv.org/abs/1512.03385)
- [A survey on deep learning in medical image analysis — Litjens et al. (2017)](https://www.sciencedirect.com/science/article/pii/S1361841517301135)
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

<div align="center">

Part of the [Machine Learning Projects](https://github.com/shsarv/Machine-Learning-Projects) collection by [Sarvesh Kumar Sharma](https://github.com/shsarv)

⭐ Star the main repo if this helped you!

</div>
