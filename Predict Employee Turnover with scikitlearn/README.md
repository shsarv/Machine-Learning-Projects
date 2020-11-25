# Predict-Employee-Turnover-with-scikit-learn

##### project on Predict Employee Churn with Decision Trees and Random Forests is divided into the following tasks:
### Task 1:Import Libraries
    Imported essential modules and helper functions from NumPy, Matplotlib, and scikit-learn.

### Task 2: Exploratory Data Analysis
    Loaded the employee dataset using pandas
    Explored the data visually by graphing various features against the target with Matplotlib.

### Task 3: Encode Categorical Features
    The dataset contains two categorical variables: Department and Salary.
    Created dummy encoded variables for both categorical variables.

### Task 4: Visualize Class Imbalance
    Used Yellowbrick's Class Balance visualizer and created a frequency plot of both classes.
    The presence or absence of a class balance problem  informed sampling strategory 
    while creating training and validation sets.

### Task 5: Create Training and Validation Sets
    Split the data into a 80/20 training/validation split.
    Used a stratified sampling strategy

### Tasks 6 & 7: Build a Decision Tree Classifier with Interactive Controls
    Used the interact function to automatically create UI controls for function arguments.
    Build and trained a decision tree classifier with scikit-learn.
    Calculated the training and validation accuracies.
    Displayed the fitted decision tree graphically.

### Task 8: Build a Random Forest Classifier with Interactive Controls
    Used the interact function again to automatically create UI controls for function arguments.
    To overcome the variance problem associated with decision trees, build and trained a random 
    forests classifier with scikit-learn.
    Calculated the training and validation accuracies.
    Displayed a fitted tree graphically.

### Task 9: Feature Importance Plots and Evaluation Metrics
    Many model forms describe the underlying impact of features relative to each other.
    Decision Tree models and Random Forest in scikit-learn, feature_importances_ attribute when fitted.
    Utilized this attribute to rank and plot the features.
