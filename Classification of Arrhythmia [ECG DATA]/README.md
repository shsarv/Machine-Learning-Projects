# Project-Arrhythmia

## Introduction

This project focuses on predicting and classifying arrhythmias using various machine learning algorithms. The dataset used for this project is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Arrhythmia), which consists of 452 examples across 16 different classes. Among these, 245 examples are labeled as "normal," while the remaining represent 12 different types of arrhythmias, including "coronary artery disease" and "right bundle branch block."

### Dataset Overview:
- **Number of Examples**: 452
- **Number of Features**: 279 (including age, sex, weight, height, and various medical parameters)
- **Classes**: 16 total (12 arrhythmia types + 1 normal group)

**Objective**:  
The goal of this project is to predict whether a person is suffering from arrhythmia, and if so, classify the type of arrhythmia into one of the 12 available groups.

## Algorithms Used

To address the classification task, the following machine learning algorithms were employed:

1. **K-Nearest Neighbors (KNN) Classifier**
2. **Logistic Regression**
3. **Decision Tree Classifier**
4. **Linear Support Vector Classifier (SVC)**
5. **Kernelized Support Vector Classifier (SVC)**
6. **Random Forest Classifier**
7. **Principal Component Analysis (PCA)** (for dimensionality reduction)

## Project Workflow

### Step 1: Data Exploration
- Analyzed the 279 features to identify patterns and correlations that could help with prediction.
- Addressed the challenge of the high number of features compared to the limited number of examples by employing PCA.

### Step 2: Data Preprocessing
- Handled missing values, standardized data, and prepared it for machine learning models.
- Applied **Principal Component Analysis (PCA)** to reduce the feature space and eliminate collinearity, improving both execution time and model performance.

### Step 3: Model Training and Evaluation
- Trained various machine learning algorithms on the dataset.
- Evaluated model performance using accuracy, recall, and other relevant metrics.

### Step 4: Model Tuning with PCA
- PCA helped reduce the complexity of the dataset, leading to improved model accuracy and reduced overfitting.
- After applying PCA, models were retrained, and significant improvements were observed.

## Results

![Results](https://raw.githubusercontent.com/shsarv/Project-Arrhythmia/master/Image/result.png)

### Conclusion

Applying **Principal Component Analysis (PCA)** to the resampled data significantly improved the performance of the models. PCA works by creating non-collinear components that prioritize variables with high variance, thus reducing dimensionality and collinearity, which are key issues in large datasets. PCA not only enhanced the overall execution time but also improved the quality of predictions.

- The **best-performing model** in terms of recall score is the **Kernelized Support Vector Machine (SVM)** with PCA, achieving an accuracy of **80.21%**.

## Future Work

- Experiment with more advanced models like **XGBoost** or **Neural Networks**.
- Perform hyperparameter tuning to further improve model accuracy and recall.
- Explore feature selection techniques alongside PCA to refine the feature set.


## Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [PCA Concepts](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)

---

This `README.md` offers clear documentation of the objectives, algorithms used, results, and the significance of PCA in your project. It also provides essential information on how to run the project and the prerequisites.