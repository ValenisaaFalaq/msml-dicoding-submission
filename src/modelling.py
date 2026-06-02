import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MANTRA SAKTI: Paksa simpan ke folder mlruns lokal
mlflow.set_tracking_uri("file:./mlruns")

def train_base_model():
    df = pd.read_csv('Workflow-CI/MLProject/dataset_preprocessing.csv')
    X = df.drop(columns=['Response']) 
    y = df['Response']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.set_experiment("Base_Model_Experiment")
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_Base"):
        logging.info("Memulai proses training model dasar...")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        logging.info(f"Training selesai. Akurasi: {acc:.4f}")

if __name__ == "__main__":
    train_base_model()