<div align="center">

# 💓 Classification of Arrhythmia — ECG Data

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20ML%20Repository-blue?style=for-the-badge)](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)
[![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-80.21%25-brightgreen?style=for-the-badge)](https://github.com/shsarv/Machine-Learning-Projects/tree/main/Classification%20of%20Arrhythmia%20%5BECG%20DATA%5D)
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](../LICENSE.md)

> Detecting the **presence or absence of cardiac arrhythmia** and classifying it into one of **16 groups** using classical ML algorithms and PCA-based dimensionality reduction on ECG signal data.

[🔙 Back to Main Repository](https://github.com/shsarv/Machine-Learning-Projects)

</div>

---

## ⚠️ Medical Disclaimer

> **This project is for educational and research purposes only.** It is not a substitute for clinical ECG interpretation or professional medical diagnosis.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [What is Arrhythmia?](#-what-is-arrhythmia)
- [Dataset](#-dataset)
- [Class Distribution](#-class-distribution)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)
- [References](#-references)

---

## 🔬 About the Project

ECG (Electrocardiogram) signals are the primary clinical tool for diagnosing heart conditions. Manual interpretation of large ECG datasets is time-consuming and error-prone. This project applies **classical ML algorithms** to automatically distinguish normal ECG readings from 15 arrhythmia subtypes using the well-known UCI Arrhythmia dataset.

A key challenge here is **high dimensionality** — 279 features with only 452 samples. The project tackles this with **PCA** and **SMOTE oversampling**, leading to significant accuracy improvements across all models.

**What this project covers:**
- Extensive EDA on a heavily imbalanced, high-dimensional tabular dataset
- Handling missing values and feature engineering from ECG signal attributes
- Dimensionality reduction with PCA
- Class imbalance handling with SMOTE oversampling
- Training and comparing 6 classifiers with and without PCA

---

## 🫀 What is Arrhythmia?

An **arrhythmia** is an irregular heartbeat — too fast, too slow, or with an irregular pattern. It is detected via ECG, which records the electrical activity of the heart. While a single arrhythmia beat may be harmless, **sustained arrhythmia can be life-threatening**, leading to stroke, heart failure, or cardiac arrest. Early automated classification is a critical tool in preventive cardiology.

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [UCI Machine Learning Repository — Arrhythmia Dataset](https://archive.ics.uci.edu/ml/datasets/Arrhythmia) |
| **Samples** | 452 patient records |
| **Features** | 279 (age, sex, weight, height + ECG signal attributes) |
| **Classes** | 16 (1 Normal + 12 Arrhythmia types + 3 unclassified groups) |
| **Missing Values** | Yes — primarily in the `J` feature column |
| **Challenge** | High dimensionality (279 features, 452 samples), severe class imbalance |

---

## 📋 Class Distribution

| Code | Class | Instances |
|:----:|-------|:---------:|
| 01 | **Normal** | 245 |
| 02 | Ischemic Changes (Coronary Artery Disease) | 44 |
| 03 | Old Anterior Myocardial Infarction | 15 |
| 04 | Old Inferior Myocardial Infarction | 15 |
| 05 | Sinus Tachycardia | 13 |
| 06 | Sinus Bradycardia | 25 |
| 07 | Ventricular Premature Contraction (PVC) | 3 |
| 08 | Supraventricular Premature Contraction | 2 |
| 09 | Left Bundle Branch Block | 9 |
| 10 | Right Bundle Branch Block | 50 |
| 11 | 1° Atrioventricular Block | 0 |
| 12 | 2° Atrioventricular Block | 0 |
| 13 | 3° Atrioventricular Block | 0 |
| 14 | Left Ventricular Hypertrophy | 4 |
| 15 | Atrial Fibrillation or Flutter | 5 |
| 16 | Others (Unclassified) | 22 |
| | **Total** | **452** |

> **Note:** 245 of 452 samples (~54%) are normal. Several arrhythmia classes have very few instances (as low as 2–3), making this a severely imbalanced multi-class problem.

---

## ⚙️ Methodology

The project follows a structured ML pipeline:

```
Raw UCI Data (452 × 279)
        │
        ▼
  Data Preprocessing
  ├── Handle missing values (median imputation)
  ├── Drop zero-variance features
  └── Encode categorical variables (sex)
        │
        ▼
  Exploratory Data Analysis
  ├── Class distribution analysis
  ├── Correlation heatmaps
  └── Feature distribution plots
        │
        ▼
  Class Imbalance Handling
  └── SMOTE Oversampling on training set
        │
        ▼
  Dimensionality Reduction
  └── PCA (retaining 95% variance)
        │
        ▼
  Model Training & Evaluation
  ├── KNN
  ├── Logistic Regression
  ├── Decision Tree
  ├── Linear SVC
  ├── Kernelized SVC  ← Best Model
  └── Random Forest
        │
        ▼
  Evaluation: Accuracy, Precision, Recall, F1-Score
```

---

## 📈 Model Performance

### Without PCA

| Model | Accuracy |
|-------|:--------:|
| KNN Classifier | ~65% |
| Logistic Regression | ~70% |
| Decision Tree | ~63% |
| Linear SVC | ~72% |
| Kernelized SVC | ~74% |
| Random Forest | ~73% |

### With PCA + SMOTE (Best Results)

| Model | Accuracy | Notes |
|-------|:--------:|-------|
| KNN Classifier | ~72% | Improved significantly |
| Logistic Regression | ~75% | Stable across classes |
| Decision Tree | ~68% | Prone to overfitting |
| Linear SVC | ~76% | Good on majority classes |
| **Kernelized SVC** ✅ | **~80.21%** | **Best recall score** |
| Random Forest | ~78% | Good overall balance |

> ✅ **Kernelized SVM with PCA** selected as the best model based on highest recall score of **80.21%**. Recall is prioritized over accuracy in medical diagnosis to minimize missed arrhythmia cases (false negatives).

---

## 🔍 Key Findings

**Why PCA helped so much:**
- With 279 features and only 452 samples, models suffered from the *curse of dimensionality*
- PCA reduces complexity by creating uncorrelated components ranked by explained variance
- It eliminates multicollinearity — a major issue when ECG signal features are highly correlated
- The resulting lower-dimensional space improves both model accuracy and training speed

**Why SMOTE was necessary:**
- Several arrhythmia classes had only 2–5 samples, making it impossible for models to learn their patterns
- SMOTE generates synthetic samples for minority classes by interpolating between existing instances
- Applied **only to training data** to prevent data leakage

**Why Kernelized SVM performed best:**
- The RBF kernel maps the PCA-transformed features into a higher-dimensional space where classes become linearly separable
- More robust to outliers than tree-based methods
- Handles the reduced but still moderately high-dimensional PCA output well

---

## 📁 Project Structure

```
Classification of Arrhythmia [ECG DATA]/
│
├── 📂 Data/
│   ├── arrhythmia.data               # Raw UCI dataset
│   └── arrhythmia.names              # Feature descriptions
│
├── 📂 Preprocessing and EDA/
│   ├── Data preprocessing.ipynb      # Missing value handling, encoding, scaling
│   └── EDA.ipynb                     # Distribution plots, correlation analysis
│
├── 📂 Model/
│   └── oversampled and pca.ipynb     # SMOTE + PCA + all model comparisons
│
├── 📂 Image/
│   └── result.png                    # Model comparison results screenshot
│
├── 📂 1- Reports and presentations/  # Project report, slides, reference papers
│
├── final with pca.ipynb              # Final consolidated notebook (main entry point)
├── requirements.txt                  # Python dependencies
└── README.md                         # You are here
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shsarv/Machine-Learning-Projects.git
cd "Machine-Learning-Projects/Classification of Arrhythmia [ECG DATA]"
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Run the notebooks in order

```bash
# Step 1 — Preprocess the data
jupyter notebook "Preprocessing and EDA/Data preprocessing.ipynb"

# Step 2 — Explore the data
jupyter notebook "Preprocessing and EDA/EDA.ipynb"

# Step 3 — Train and evaluate all models
jupyter notebook "final with pca.ipynb"
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.7+ |
| ML Library | scikit-learn |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Dimensionality Reduction | PCA (scikit-learn) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |

---

## 📚 References

- [UCI ML Repository — Arrhythmia Dataset](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)
- Guvenir, H.A., et al. (1997). *A Supervised Machine Learning Algorithm for Arrhythmia Analysis.* Computers in Cardiology.
- [imbalanced-learn SMOTE Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)

---

<div align="center">

Part of the [Machine Learning Projects](https://github.com/shsarv/Machine-Learning-Projects) collection by [Sarvesh Kumar Sharma](https://github.com/shsarv)

⭐ Star the main repo if this helped you!

</div>
