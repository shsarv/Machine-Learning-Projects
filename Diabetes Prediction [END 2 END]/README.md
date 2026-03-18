<div align="center">

# 🩺 Diabetes Prediction — End to End

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Model-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](../LICENSE.md)

> A full **end-to-end machine learning web application** that predicts the likelihood of diabetes in a patient based on key health diagnostics — from model training to a live Flask deployment.

[🔙 Back to Main Repository](https://github.com/shsarv/Machine-Learning-Projects)

</div>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Dataset](#-dataset)
- [Features Used](#-features-used)
- [Model & Performance](#-model--performance)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [App Screenshots](#-app-screenshots)
- [Tech Stack](#-tech-stack)

---

## 🧠 About the Project

Diabetes is one of the most prevalent chronic diseases worldwide, and early detection significantly improves patient outcomes. This project builds a **binary classification model** to predict whether a patient is likely to have diabetes based on diagnostic measurements, and wraps it in an interactive **Flask web application** so anyone can get a prediction by entering their health values.

**What this project covers:**
- Exploratory data analysis (EDA) and data preprocessing
- Feature engineering and handling class imbalance
- Training and comparing multiple ML classifiers
- Serializing the best model with `pickle`
- Building and deploying a Flask web app with a clean UI

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | Pima Indians Diabetes Dataset |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) / UCI ML Repository |
| **Samples** | 768 patients |
| **Features** | 8 numeric diagnostic features |
| **Target** | Binary — `1` (Diabetic) / `0` (Non-Diabetic) |
| **Class Balance** | ~65% Non-Diabetic · ~35% Diabetic |

---

## 🔬 Features Used

| Feature | Description |
|---------|-------------|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (µU/ml) |
| `BMI` | Body mass index (weight in kg / height in m²) |
| `DiabetesPedigreeFunction` | Likelihood of diabetes based on family history |
| `Age` | Age in years |

---

## 🤖 Model & Performance

Multiple classifiers were trained and evaluated. The best-performing model was selected for deployment.

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression | ~77% | ~74% | ~67% | ~70% |
| K-Nearest Neighbors | ~74% | ~70% | ~63% | ~66% |
| Support Vector Machine | ~78% | ~75% | ~68% | ~71% |
| Decision Tree | ~73% | ~68% | ~65% | ~66% |
| **Random Forest** ✅ | **~81%** | **~78%** | **~72%** | **~75%** |
| Gradient Boosting | ~80% | ~76% | ~71% | ~73% |

> ✅ **Random Forest** selected as the final model based on highest overall accuracy and F1-score.

**Preprocessing steps:**
- Replaced biologically implausible zero values (e.g., `Glucose = 0`) with feature medians
- Scaled features using `StandardScaler`
- Split data: 80% train / 20% test with stratification

---

## 📁 Project Structure

```
Diabetes Prediction [END 2 END]/
│
├── 📂 Dataset/
│   └── diabetes.csv              # Pima Indians Diabetes dataset
│
├── 📂 Model/
│   └── diabetes_model.pkl        # Serialized trained model (pickle)
│
├── 📂 notebooks/
│   └── diabetes_prediction.ipynb # EDA, training, and evaluation notebook
│
├── 📂 static/
│   └── css/
│       └── style.css             # App styling
│
├── 📂 templates/
│   ├── index.html                # Home / input form
│   └── result.html               # Prediction result page
│
├── app.py                        # Flask application entry point
├── requirements.txt              # Python dependencies
└── README.md                     # You are here
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shsarv/Machine-Learning-Projects.git
cd "Machine-Learning-Projects/Diabetes Prediction [END 2 END]"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask app

```bash
python app.py
```

Open your browser and navigate to → **http://127.0.0.1:5000**

### 5. (Optional) Re-train the model

Open the Jupyter notebook to explore the data and retrain from scratch:

```bash
jupyter notebook notebooks/diabetes_prediction.ipynb
```

---

## 📸 App Screenshots

> The web app presents a clean form where users input their health metrics and receive an instant prediction.

| Input Form | Prediction Result |
|:----------:|:-----------------:|
| User enters 8 health parameters | App displays **Diabetic** or **Not Diabetic** with confidence |

![](https://github.com/shsarv/Machine-Learning-Projects/blob/main/Diabetes%20Prediction%20%5BEND%202%20END%5D/Diabetes-prediction%20deployed/Resource/live1.gif)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.7+ |
| ML Library | scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Frontend | HTML5, CSS3, Bootstrap |
| Model Serialization | Pickle |
| Notebook | Jupyter |

---

## 📚 References

- [Pima Indians Diabetes Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

<div align="center">

Part of the [Machine Learning Projects](https://github.com/shsarv/Machine-Learning-Projects) collection by [Sarvesh Kumar Sharma](https://github.com/shsarv)

⭐ Star the main repo if this helped you!

</div>
