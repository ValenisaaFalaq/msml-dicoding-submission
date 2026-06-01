import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    logging.info(f"Memuat data dari {filepath}")
    return pd.read_csv(filepath)

def train_base_model():
    # 1. Siapkan Data
    df = load_data('./data/processed/dataset_clean.csv')
    
    # Pisahkan fitur (X) dan target (y). Sesuaikan 'Response' dengan nama kolom targetmu jika berbeda (huruf besar/kecil).
    X = df.drop(columns=['Response']) 
    y = df['Response']
    
    # Split data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    mlflow.set_experiment("Base_Model_Experiment")
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_Base"):
        logging.info("Memulai proses training model dasar...")
        
        # Inisialisasi dan latih model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        logging.info(f"Training selesai. Akurasi Base Model: {acc:.4f}")
        logging.info("Hasil eksperimen telah dicatat oleh MLflow autolog.")

if __name__ == "__main__":
    train_base_model()