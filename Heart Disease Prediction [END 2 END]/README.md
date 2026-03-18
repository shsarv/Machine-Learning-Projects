- Look for final Project At **![https://github.com/shsarv/Cardio-Monitor](https://github.com/shsarv/Cardio-Monitor)**

<div align="center">

# 🫀 Cardio Monitor — Heart Disease Prediction Web App

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Database-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-92%25-brightgreen?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](LICENSE)

> **Cardio Monitor** is a full-stack web application that predicts whether a patient is at risk of developing **heart disease** using a machine learning model with **92% accuracy** — built with Flask, MongoDB, and scikit-learn. Course project for **Big Data Analytics (BCSE0158)**.

[![Stars](https://img.shields.io/github/stars/shsarv/Cardio-Monitor?style=social)](https://github.com/shsarv/Cardio-Monitor/stargazers)
[![Forks](https://img.shields.io/github/forks/shsarv/Cardio-Monitor?style=social)](https://github.com/shsarv/Cardio-Monitor/forks)

[🔗 Core ML Project](https://github.com/shsarv/Heart-Disease-Prediction) &nbsp;·&nbsp; [🐛 Report Bug](https://github.com/shsarv/Cardio-Monitor/issues) &nbsp;·&nbsp; [✨ Request Feature](https://github.com/shsarv/Cardio-Monitor/issues)

</div>

---

## ⚠️ Medical Disclaimer

> **This application is for educational and research purposes only.** It does not constitute medical advice. Always consult a qualified cardiologist or medical professional for clinical decisions.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [How It Works](#-how-it-works)
- [Dataset & Features](#-dataset--features)
- [Model & Performance](#-model--performance)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Future Roadmap](#-future-roadmap)
- [Tech Stack](#-tech-stack)
- [References](#-references)

---

## 🔬 About the Project

Heart disease is the leading cause of death globally. Early detection through continuous monitoring can significantly reduce mortality rates. **Cardio Monitor** combines:

- A **machine learning classifier** (92% accuracy) trained on the Cleveland Heart Disease dataset
- A **Flask web app** for real-time patient input and prediction
- A **MongoDB** backend for storing patient records and prediction history
- A **visualization module** for EDA and model insights
- A roadmap toward **Apache Spark Streaming** for large-scale real-time data processing

The core ML research and model building is documented in the companion repository: [shsarv/Heart-Disease-Prediction](https://github.com/shsarv/Heart-Disease-Prediction).

---

## ⚙️ How It Works

```
Patient Inputs Clinical Data via Web Form
              │
              ▼
       Flask (app.py)
       routes request to
              │
              ▼
     prediction.py
     Loads Heart_model1.pkl
     Runs model.predict()
              │
       ┌──────┴──────┐
       ▼             ▼
   At Risk ❤️‍🩹    Not at Risk ✅
       │
       ▼
  Result displayed on web page
  Record saved to MongoDB (database.py)
```

---

## 📊 Dataset & Features

| Property | Details |
|----------|---------|
| **File** | `heart.csv` |
| **Source** | Cleveland Heart Disease Dataset (UCI ML Repository) |
| **Samples** | 303 patient records |
| **Task** | Binary classification — Heart Disease (1) / No Heart Disease (0) |

### Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| `age` | Age of patient | Years |
| `sex` | Sex | 0 = Female, 1 = Male |
| `cp` | Chest pain type | 0–3 |
| `trestbps` | Resting blood pressure | mm Hg |
| `chol` | Serum cholesterol | mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl | 0 / 1 |
| `restecg` | Resting ECG results | 0–2 |
| `thalach` | Maximum heart rate achieved | bpm |
| `exang` | Exercise induced angina | 0 / 1 |
| `oldpeak` | ST depression induced by exercise | Float |
| `slope` | Slope of peak exercise ST segment | 0–2 |
| `ca` | Number of major vessels coloured by fluoroscopy | 0–3 |
| `thal` | Thalassemia | 0–3 |
| `target` ⭐ | **Heart disease present** | 0 / 1 |

---

## 🤖 Model & Performance

| Metric | Value |
|--------|:-----:|
| **Accuracy** | **92%** |
| **Saved Model** | `Heart_model1.pkl` / `heartmodel.pkl` |
| **Algorithm** | scikit-learn classifier (see core project) |
| **Library** | scikit-learn + mlxtend |

> Two model files are present in the repo: `Heart_model1.pkl` (primary, used by `prediction.py`) and `heartmodel.pkl` (earlier iteration). Both are serialized with `pickle`.

> For full model building details — EDA, feature selection, algorithm comparison, and evaluation — see the core project: [shsarv/Heart-Disease-Prediction](https://github.com/shsarv/Heart-Disease-Prediction).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              Flask Application              │
│                  (app.py)                   │
│                                             │
│  ┌──────────┐  ┌────────────┐  ┌─────────┐ │
│  │templates/│  │prediction  │  │database │ │
│  │  HTML    │  │   .py      │  │  .py    │ │
│  │  pages   │  │ ML model   │  │ MongoDB │ │
│  └──────────┘  └────────────┘  └─────────┘ │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │          static/                     │   │
│  │   CSS · JS · images                  │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
  Heart_model1.pkl       MongoDB Atlas
  (scikit-learn)         (patient records
                         + predictions)
```

---

## 📁 Project Structure

```
Cardio-Monitor/
│
├── 📂 heart disease prediction/     # Jupyter notebooks — EDA & model training
├── 📂 static/                       # CSS, JS, images
├── 📂 templates/                    # Jinja2 HTML templates (input form, result pages)
├── 📂 __pycache__/
│
├── app.py                           # Flask entry point — routes and app config
├── prediction.py                    # Loads Heart_model1.pkl, runs inference
├── modelbuild.py                    # Model training and serialization script
├── database.py                      # MongoDB connection and CRUD operations
├── visualization.py                 # EDA and data visualization utilities
│
├── Heart_model1.pkl                 # Primary trained model (pickle)
├── heartmodel.pkl                   # Alternate model iteration (pickle)
├── heart.csv                        # Cleveland Heart Disease dataset
├── Input Data.png                   # Screenshot of the web app input form
│
├── Procfile                         # Heroku deployment config
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- MongoDB (local or [MongoDB Atlas](https://www.mongodb.com/cloud/atlas))

### 1. Clone the repository

```bash
git clone https://github.com/shsarv/Cardio-Monitor.git
cd Cardio-Monitor
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Configure MongoDB

In `database.py`, update your MongoDB connection string:

```python
# Local MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")

# MongoDB Atlas (cloud)
client = pymongo.MongoClient("mongodb+srv://<user>:<password>@cluster.mongodb.net/")
```

### 4. Run the app

```bash
python app.py
```

Navigate to → **http://127.0.0.1:5000**

### 5. Deploy to Heroku

```bash
heroku login
heroku create cardio-monitor-app
git push heroku main
heroku open
```

> The `Procfile` already contains: `web: gunicorn app:app`

---

## 🗺️ Future Roadmap

| Feature | Status |
|---------|:------:|
| Flask web app with MongoDB | ✅ Done |
| 92% accuracy ML model | ✅ Done |
| Heroku deployment | ✅ Done |
| **Apache Spark Streaming** — real-time patient data ingestion | 🔜 Planned |
| **PySpark MLlib** — large-scale distributed model training | 🔜 Planned |
| **Deep Learning model** (Keras/TensorFlow) | 🔜 Planned |
| Live demo deployment | 🔜 Planned |

---

## 🛠️ Tech Stack

**Current:**

| Layer | Technology |
|-------|-----------|
| Language | Python 3.7+ |
| Web Framework | Flask |
| ML Library | scikit-learn, mlxtend |
| Database | MongoDB (PyMongo) |
| Model Serialization | Pickle |
| Frontend | HTML5, CSS3, Bootstrap |
| Deployment | Heroku (Procfile + gunicorn) |
| Notebook | Jupyter |

**Planned (Future):**

| Layer | Technology |
|-------|-----------|
| Streaming | Apache Spark Streaming |
| Distributed ML | PySpark MLlib |
| Deep Learning | Keras / TensorFlow (DeepL) |
| Database (scale) | MongoDB Atlas |

---

## 📚 References

- [Cleveland Heart Disease Dataset — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- [Core ML Project — shsarv/Heart-Disease-Prediction](https://github.com/shsarv/Heart-Disease-Prediction)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [PyMongo Documentation](https://pymongo.readthedocs.io/)
- [mlxtend Documentation](https://rasbt.github.io/mlxtend/)
- [Apache Spark Streaming](https://spark.apache.org/streaming/)

---

<div align="center">

**Created by [Sarvesh Kumar Sharma](https://github.com/shsarv)**

Course Project — Big Data Analytics (BCSE0158)

⭐ Star this repo if you found it helpful!

</div>
