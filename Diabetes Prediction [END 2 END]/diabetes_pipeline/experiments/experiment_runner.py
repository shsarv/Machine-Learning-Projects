# diabetes_pipeline/experiments/experiment_runner.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from diabetes_pipeline.data_preprocessing import load_and_preprocess

X_train, X_test, y_train, y_test = load_and_preprocess()

models = {
	"LogisticRegression": LogisticRegression(max_iter=1000),
	"DecisionTree": DecisionTreeClassifier(random_state=0),
	"RandomForest": RandomForestClassifier(n_estimators=50, random_state=0),
	"SVM": SVC()
}

results = []

for name, model in models.items():
	pipeline = Pipeline([
		("scaler", StandardScaler()),
		("model", model)
	])

	pipeline.fit(X_train, y_train)
	preds = pipeline.predict(X_test)

	results.append({
		"Model": name,
		"Accuracy": accuracy_score(y_test, preds),
		"F1 Score": f1_score(y_test, preds)
	})

df = pd.DataFrame(results)
print(df)

df.to_csv("diabetes_pipeline/experiments/results.csv", index=False)
