import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import mlflow
import joblib
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Pastikan folder artifacts tersedia
os.makedirs("src/artifacts", exist_ok=True)
os.makedirs("models", exist_ok=True)

def train_and_tune():
    # 1. Load Data
    df = pd.read_csv('./data/processed/dataset_clean.csv')
    X = df.drop(columns=['Response'])
    y = df['Response']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Definisikan Parameter Grid
    # Kita gunakan GridSearchCV agar pencarian sistematis dan optimal.
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

    # 3. Inisialisasi Model & Tuning
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    
    logging.info("Memulai Hyperparameter Tuning dengan GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    logging.info(f"Parameter terbaik ditemukan: {best_params}")

    # 4. Evaluasi Model Terbaik
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 5. Buat Artifacts (Wajib untuk Advance)
    # A. Classification Report
    report = classification_report(y_test, y_pred)
    report_path = "src/artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # B. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Tuned Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_path = "src/artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # C. Simpan Model Fisik
    model_dir = "models/mlflow_model"
    
    # Hapus folder model lama jika sudah ada agar tidak error saat ditimpa
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        
    # Simpan model dengan format MLflow (folder berisi MLmodel, conda.yaml, dll)
    mlflow.sklearn.save_model(best_model, model_dir)

    # 6. Manual Logging ke MLflow (TANPA AUTOLOG)
    mlflow.set_experiment("Tuned_Model_Experiment")
    with mlflow.start_run(run_name="RF_GridSearch_Manual"):
        logging.info("Mencatat ke MLflow secara manual...")
        
        # Log Parameters Wajib
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("search_method", "GridSearchCV")
        mlflow.log_param("best_n_estimators", best_params['n_estimators'])
        mlflow.log_param("best_max_depth", best_params['max_depth'])
        mlflow.log_param("best_min_samples_split", best_params['min_samples_split'])
        mlflow.log_param("random_state", 42)

        # Log Metrics Wajib
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log Artifacts Wajib (Gambar dan Teks)
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(cm_path)
        
        # Log Model
        mlflow.sklearn.log_model(best_model, "random_forest_model")

        logging.info("Eksperimen Tuning berhasil dicatat di MLflow!")

if __name__ == "__main__":
    train_and_tune()