import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import mlflow
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MANTRA SAKTI: Paksa simpan ke folder mlruns lokal
mlflow.set_tracking_uri("file:./mlruns")

os.makedirs("src/artifacts", exist_ok=True)
os.makedirs("src/models", exist_ok=True)

def train_and_tune():
    df = pd.read_csv('Workflow-CI/MLProject/dataset_preprocessing.csv')
    X = df.drop(columns=['Response'])
    y = df['Response']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {'n_estimators': [50], 'max_depth': [10], 'min_samples_split': [2]}
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=2, n_jobs=-1)
    
    logging.info("Memulai Hyperparameter Tuning...")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    report_path = "src/artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, y_pred))

    cm_path = "src/artifacts/confusion_matrix.png"
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.savefig(cm_path)
    plt.close()

    mlflow.set_experiment("Tuned_Model_Experiment")
    with mlflow.start_run(run_name="RF_GridSearch_Manual"):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        logging.info("Tuning berhasil dicatat di MLflow!")

if __name__ == "__main__":
    train_and_tune()