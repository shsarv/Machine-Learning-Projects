<div align="center">

# 🏃 Human Activity Recognition — 2D Pose + LSTM RNN

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![LSTM](https://img.shields.io/badge/LSTM-2%20Stacked%20Layers-9B59B6?style=for-the-badge)]()
[![Accuracy](https://img.shields.io/badge/Accuracy->90%25-brightgreen?style=for-the-badge)]()
[![ngrok](https://img.shields.io/badge/Deployed-ngrok-1F8ACB?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](../LICENSE.md)

> Classifies **6 human activities** from **2D pose time series** (OpenPose keypoints) using a **2-layer stacked LSTM RNN** built in TensorFlow 1.x — achieving **>90% accuracy** in ~7 minutes of training. Deployed via ngrok with a Flask web app and `sample_video.mp4` demo.

[🔙 Back to Main Repository](https://github.com/shsarv/Machine-Learning-Projects)

</div>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Key Idea — Why 2D Pose?](#-key-idea--why-2d-pose)
- [Dataset](#-dataset)
- [LSTM Architecture](#-lstm-architecture)
- [Training Configuration](#-training-configuration)
- [Results & Findings](#-results--findings)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [References](#-references)

---

## 🔬 About the Project

This experiment classifies human activities using **2D pose time series data** and a **stacked LSTM RNN**. Rather than feeding raw RGB images or expensive 3D pose data into the network, it uses **2D (x, y) keypoints** extracted from video frames via OpenPose — a much lighter and more accessible input representation.

The core research questions:

- Can **2D pose** match **3D pose** accuracy for activity recognition? (removes need for RGBD cameras)
- Can **2D pose** match **raw RGB image** accuracy? (smaller input = smaller model = better with limited data)
- Does this approach generalize to **animal** behaviour classification for robotics applications?

The network architecture is based on Guillaume Chevalier's *LSTMs for Human Activity Recognition (2016)*, with key modifications for large class-ordered datasets using **random batch sampling without replacement**.

---

## 🧠 Key Idea — Why 2D Pose?

```
Raw Video Frame (640×480 RGB)
        │
        ▼
   OpenPose Inference
   18 body keypoints × (x, y) coords
        │
        ▼
   36-dimensional feature vector per frame
        │
        ▼  (32 frames = 1 time window)
   LSTM RNN  →  Activity Class
```

| Input Type | Pros | Cons |
|------------|------|------|
| Raw RGB images | High information | Large models, lots of data needed |
| 3D pose (RGBD) | Rich spatial info | Needs depth sensors |
| **2D pose (x,y)** ✅ | Lightweight, RGB-only camera, small model | Some spatial ambiguity |

> Limiting the feature vector to 2D pose keypoints allows for a **smaller LSTM model** that generalises better on limited datasets — particularly relevant for future animal behaviour recognition tasks.

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | Berkeley Multimodal Human Action Database (MHAD) — 2D poses extracted via OpenPose |
| **Download** | `RNN-HAR-2D-Pose-database.zip` (~19.2 MB, Google Drive) |
| **Subjects** | 12 |
| **Angles** | 4 camera angles |
| **Repetitions** | 5 per subject per action |
| **Total videos** | 1,438 (2 missing from original 1,440) |
| **Total frames** | 211,200 |
| **Training windows** | 22,625 (32 timesteps each, 50% overlap) |
| **Test windows** | 5,751 |
| **Input shape** | `(22625, 32, 36)` → windows × timesteps × features |
| **Preprocessing** | ❌ None — raw, unnormalized pose coordinates |

### Activity Classes (6)

| Label | Activity |
|-------|----------|
| `JUMPING` | Vertical jumps |
| `JUMPING_JACKS` | Jumping jacks |
| `BOXING` | Boxing motions |
| `WAVING_2HANDS` | Waving with both hands |
| `WAVING_1HAND` | Waving with one hand |
| `CLAPPING_HANDS` | Clapping hands |

### Data Files

```
RNN-HAR-2D-Pose-database/
├── X_train.txt    # 22,625 training windows (36 comma-separated floats per row)
├── X_test.txt     # 5,751 test windows
├── Y_train.txt    # Training labels (0–5)
└── Y_test.txt     # Test labels (0–5)
```

---

## 🏗️ LSTM Architecture

```
Input: (batch_size, 32 timesteps, 36 features)
             │
             ▼
  Linear projection: 36 → 34 (ReLU)
             │
             ▼
  ┌──────────────────────────────────┐
  │  BasicLSTMCell(34, forget_bias=1)│  ← Layer 1
  ├──────────────────────────────────┤
  │  BasicLSTMCell(34, forget_bias=1)│  ← Layer 2
  └──────────────────────────────────┘
  tf.contrib.rnn.MultiRNNCell (stacked)
  tf.contrib.rnn.static_rnn (many-to-one)
             │
        Last output only
             │
             ▼
  Linear: 34 → 6
  Softmax → Activity class
```

> **Why n_hidden = 34?** Testing across a range of hidden unit counts showed best generalisation when hidden units ≈ n_input (36). 34 was found to be optimal.

> **Many-to-one classifier** — only the last LSTM output (timestep 32) is used for classification, not the full sequence output.

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | TensorFlow 1.x (`%tensorflow_version 1.x`) |
| Timesteps (`n_steps`) | 32 |
| Input features (`n_input`) | 36 (18 keypoints × x, y) |
| Hidden units (`n_hidden`) | 34 |
| Classes (`n_classes`) | 6 |
| Epochs | 300 |
| Batch size | 512 |
| Optimizer | Adam |
| Initial learning rate | 0.005 |
| LR decay | Exponential — `0.96` per 100,000 steps |
| Loss | Softmax cross-entropy + L2 regularization |
| L2 lambda | 0.0015 |
| Batch strategy | Random sampling **without replacement** (prevents class-order bias) |
| Training time | ~7 minutes (Google Colab) |

**L2 regularization formula:**
```python
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)
cost = tf.reduce_mean(softmax_cross_entropy) + l2
```

**Decayed learning rate:**
```python
learning_rate = init_lr * decay_rate ^ (global_step / decay_steps)
# = 0.005 * 0.96 ^ (global_step / 100000)
```

---

## 📈 Results & Findings

| Metric | Value |
|--------|:-----:|
| **Final Accuracy** | **> 90%** |
| Training time | ~7 minutes |

**Confusion pairs observed:**
- `CLAPPING_HANDS` ↔ `BOXING` — similar upper-body motion pattern
- `JUMPING_JACKS` ↔ `WAVING_2HANDS` — symmetric arm movements

**Key conclusions:**
- 2D pose achieves >90% accuracy, validating its use over more expensive 3D pose or raw RGB inputs
- Hidden units ≈ n_input (34 ≈ 36) gives optimal generalisation
- Random batch sampling without replacement is **critical** — ordered class batches degrade training significantly
- Approach is promising for future animal behaviour estimation with autonomous mobile robots

---

## 📁 Project Structure

```
Human Activity Detection/
│
├── 📂 images/                                        # Result plots and visualizations
├── 📂 models/                                        # Saved LSTM model weights
├── 📂 src/                                           # Helper source scripts
├── 📂 templates/                                     # HTML templates (Flask app)
│
├── Human_Activity_Recogination.ipynb                 # Main notebook — dataset, LSTM, training
├── Human_Action_Classification_deployment_with_ngrok.ipynb  # Flask + ngrok deployment notebook
├── lstm_train.ipynb                                  # Standalone LSTM training notebook
├── app.py                                            # Flask web application
├── sample_video.mp4                                  # Sample video for live demo
└── requirements.txt                                  # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shsarv/Machine-Learning-Projects.git
cd "Machine-Learning-Projects/Human Activity Detection"
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

> ⚠️ **TensorFlow 1.x required.** The LSTM uses `tf.contrib.rnn` and `tf.placeholder` APIs from TF1.
> ```bash
> pip install tensorflow==1.15.0
> ```

### 3. Download the dataset

The dataset is downloaded automatically in the notebook:
```python
!wget -O RNN-HAR-2D-Pose-database.zip \
  https://drive.google.com/u/1/uc?id=1IuZlyNjg6DMQE3iaO1Px6h1yLKgatynt
!unzip RNN-HAR-2D-Pose-database.zip
```

### 4. Run on Google Colab (recommended)

```
1. Open Human_Activity_Recogination.ipynb in Google Colab
2. Runtime → Change runtime type → GPU (optional, speeds training)
3. Run all cells — training completes in ~7 minutes
```

### 5. Deploy with ngrok

```
Open Human_Action_Classification_deployment_with_ngrok.ipynb
Follow the ngrok setup cells to expose the Flask app publicly
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.7+ |
| Deep Learning | TensorFlow 1.x (`tf.contrib.rnn`) |
| Model | 2-layer stacked LSTM (`BasicLSTMCell`) |
| Pose Extraction | OpenPose (CMU Perceptual Computing Lab) |
| Data Processing | NumPy |
| Visualization | Matplotlib |
| Web Framework | Flask |
| Deployment | ngrok (tunnel) |
| Notebook | Jupyter / Google Colab |

---

## 📚 References

- Guillaume Chevalier (2016). *LSTMs for Human Activity Recognition.* [github.com/guillaume-chevalier](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition) — MIT License
- [Berkeley MHAD Dataset](http://tele-immersion.citris-uc.org/berkeley_mhad)
- [OpenPose — CMU Perceptual Computing Lab](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- Goodfellow et al. *"It has been observed in practice that when using a larger batch there is a significant degradation in the quality of the model..."* — basis for small batch strategy
- [Andrej Karpathy — The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) — referenced for many-to-one classifier design

---

<div align="center">

Part of the [Machine Learning Projects](https://github.com/shsarv/Machine-Learning-Projects) collection by [Sarvesh Kumar Sharma](https://github.com/shsarv)

⭐ Star the main repo if this helped you!

</div>
