# Diabetes Prediction

This repository contains an end-to-end machine learning project aimed at predicting the likelihood of diabetes based on user-provided health data. The project demonstrates the full machine learning pipeline from data gathering to model deployment using a Flask web application hosted on Heroku.

## Project Overview

The goal of this project is to create a seamless process for predicting diabetes by building a machine learning model that analyzes various health parameters. The web application takes user input, processes the data through the model, and provides the prediction result on a new page.

## Project Objectives

The project follows these key steps:

1. **Data Gathering**: Collected relevant medical data from various sources, including public datasets.
2. **Descriptive Analysis**: Explored the dataset to understand the underlying patterns and trends.
3. **Data Visualizations**: Created insightful visualizations to represent key relationships in the data.
4. **Data Preprocessing**: Cleaned and transformed the data for use in the machine learning model.
5. **Data Modelling**: Trained a machine learning model using scikit-learn to predict diabetes.
6. **Model Evaluation**: Assessed the model's performance using various metrics to ensure accuracy.
7. **Model Deployment**: Deployed the model as a web application using Flask, hosted on Heroku.

## Technical Aspects

### Machine Learning Model
- **Library**: scikit-learn
- **Algorithms Used**: Logistic Regression, Decision Trees, Random Forests (or any chosen algorithms based on your project)
- **Input Features**: The following fields are taken from the user:
  - Number of Pregnancies
  - Insulin Level
  - Age
  - Body Mass Index (BMI)
  - Blood Pressure
  - Glucose Level
  - Skin Thickness
  - Diabetes Pedigree Function
- **Output**: The model predicts whether the person is likely to have diabetes (Yes/No).

### Web Application
- **Framework**: Flask
- **Deployment**: Hosted on Heroku for easy access.
- **Functionality**: 
  - The user provides health-related data via a form.
  - After submitting the form, the model processes the data and presents the prediction on a new page.
  
## How to Use

### Prerequisites
- Python 3.x
- Flask
- scikit-learn
- Pandas
- Heroku CLI (for deployment)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/shsarv/Machine-learning-projects.git
   cd diabetes-prediction-[End-2-END]/Diabetes-prediction-deployed
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open your browser and go to `http://localhost:5000` to access the web app.

### Deployment on Heroku

To deploy the app on Heroku, follow these steps:

1. Login to Heroku:
   ```bash
   heroku login
   ```

2. Create a new Heroku app:
   ```bash
   heroku create your-app-name
   ```

3. Push your code to Heroku:
   ```bash
   git push heroku main
   ```

4. Open the app in your browser:
   ```bash
   heroku open
   ```

## Future Enhancements

- Add more advanced machine learning models for improved prediction accuracy.
- Implement user authentication for a more personalized experience.
- Improve UI/UX for better usability.
- Integrate more health-related data for broader insights.

## Contributing

Feel free to contribute by submitting issues or pull requests. For major changes, please open an issue first to discuss what you'd like to change.

## Acknowledgments

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Heroku Documentation](https://devcenter.heroku.com/)

---