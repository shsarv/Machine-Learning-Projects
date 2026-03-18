<div align="center">

# 🎓 Getting Admission in College Prediction

[![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/mohansacharya/graduate-admissions)
[![Best R²](https://img.shields.io/badge/Best%20R²-0.821-brightgreen?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](../LICENSE.md)

> Predicts a student's **probability of graduate college admission** (as a continuous value between 0 and 1) from 7 academic and profile features — using a `GridSearchCV`-powered model comparison across 6 regression algorithms.

[🔙 Back to Main Repository](https://github.com/shsarv/Machine-Learning-Projects)

</div>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Dataset](#-dataset)
- [Features](#-features)
- [Methodology](#-methodology)
- [Model Comparison Results](#-model-comparison-results)
- [Final Model Performance](#-final-model-performance)
- [Sample Predictions](#-sample-predictions)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)

---

## 🔬 About the Project

Getting into a good graduate program is one of the most competitive processes for students worldwide. This project builds a **regression model** that predicts the probability of admission based on a student's GRE score, TOEFL score, CGPA, university rating, SOP, LOR, and research experience.

Six regression algorithms are trained and compared using **GridSearchCV with 5-fold cross-validation** via a custom `find_best_model()` function. The best-performing model is then evaluated on a held-out test set.

**What this project covers:**
- Exploratory data analysis on 500 graduate applicant profiles
- Custom `find_best_model()` with GridSearchCV across 6 regressors
- Feature importance and correlation analysis
- Linear Regression selected as the final model with **R² = 0.821** on test set

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **File** | `admission_predict.csv` |
| **Source** | [Kaggle — Graduate Admissions](https://www.kaggle.com/mohansacharya/graduate-admissions) |
| **Rows** | 500 student records |
| **Columns** | 9 (including Serial No. and target) |
| **Task** | Regression — predict `Chance of Admit` ∈ [0, 1] |
| **Missing Values** | None |

---

## 🔬 Features

| Column | Type | Range | Description |
|--------|------|:-----:|-------------|
| `GRE Score` | Integer | 290–340 | Graduate Record Examination score |
| `TOEFL Score` | Integer | 92–120 | Test of English as a Foreign Language score |
| `University Rating` | Integer | 1–5 | Prestige rating of undergraduate university |
| `SOP` | Float | 1.0–5.0 | Strength of Statement of Purpose |
| `LOR` | Float | 1.0–5.0 | Strength of Letter of Recommendation |
| `CGPA` | Float | 6.8–9.92 | Undergraduate GPA (out of 10) |
| `Research` | Binary | 0 / 1 | Research experience (0 = No, 1 = Yes) |
| `Chance of Admit` ⭐ | Float | 0.34–0.97 | **Target variable** — probability of admission |

> `Serial No.` is dropped before training as it carries no predictive information.

---

## ⚙️ Methodology

```
Load admission_predict.csv (500 × 9)
          │
          ▼
EDA + Correlation Analysis
(heatmap, pairplots, distributions)
          │
          ▼
Drop 'Serial No.' column
Define X (7 features) and y ('Chance of Admit')
          │
          ▼
find_best_model(X, y)
└── GridSearchCV (cv=5) over 6 models
          │
          ▼
Select best model → Linear Regression (normalize=True)
          │
          ▼
Train/Test Split (80/20, random_state=5)
→ 400 train samples, 100 test samples
          │
          ▼
Fit LinearRegression(normalize=True)
Evaluate on test set → R² = 0.821
          │
          ▼
Sample Predictions
```

---

## 📈 Model Comparison Results

All 6 models evaluated using `GridSearchCV(cv=5)` via the custom `find_best_model()` function:

| Model | Best Parameters | CV R² Score |
|-------|----------------|:-----------:|
| **Linear Regression** ✅ | `{'normalize': True}` | **0.8108** |
| Random Forest | `{'n_estimators': 15}` | 0.7689 |
| KNN | `{'n_neighbors': 20}` | 0.7230 |
| SVR | `{'gamma': 'scale'}` | 0.6541 |
| Decision Tree | `{'criterion': 'mse', 'splitter': 'random'}` | 0.5868 |
| Lasso | `{'alpha': 1, 'selection': 'random'}` | 0.2151 |

> ✅ **Linear Regression** selected as the final model — highest cross-validation R² score of **0.8108**.

> Lasso performed poorly (R² = 0.2151) because L1 regularization shrinks coefficients aggressively, which is harmful here where all 7 features are genuinely correlated with admission probability.

---

## 🏆 Final Model Performance

| Metric | Value |
|--------|:-----:|
| Model | `LinearRegression(normalize=True)` |
| 5-Fold Cross-Validation Score | **81.0%** |
| Train samples | 400 |
| Test samples | 100 |
| **Test R² Score** | **0.8215** |

---

## 🔮 Sample Predictions

```python
# Input: [GRE, TOEFL, Univ Rating, SOP, LOR, CGPA, Research]

model.predict([[337, 118, 4, 4.5, 4.5, 9.65, 0]])
# → Chance of getting into UCLA is 92.855%

model.predict([[320, 113, 2, 2.0, 2.5, 8.64, 1]])
# → Chance of getting into UCLA is 73.627%
```

---

## 📁 Project Structure

```
Getting Admission in College Prediction/
│
├── Admission_prediction.ipynb      # Main notebook — EDA, model comparison, training
├── admission_predict.csv           # Dataset (500 student records)
├── requirements.txt                # Python dependencies
└── README.md                       # You are here
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shsarv/Machine-Learning-Projects.git
cd "Machine-Learning-Projects/Getting Admission in College Prediction"
```

### 2. Set up environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook Admission_prediction.ipynb
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.7.4 |
| ML Library | scikit-learn |
| Model Selection | `GridSearchCV`, `cross_val_score` |
| Models | `LinearRegression`, `Lasso`, `SVR`, `DecisionTreeRegressor`, `RandomForestRegressor`, `KNeighborsRegressor` |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Notebook | Jupyter |

---

<div align="center">

Part of the [Machine Learning Projects](https://github.com/shsarv/Machine-Learning-Projects) collection by [Sarvesh Kumar Sharma](https://github.com/shsarv)

⭐ Star the main repo if this helped you!

</div>
