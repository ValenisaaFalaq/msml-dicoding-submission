import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    logging.info(f"Memuat data dari {file_path}")
    # Sesuaikan pemisah (separator) jika file aslimu menggunakan koma, ganti '\t' jadi ','
    return pd.read_csv(file_path, sep='\t')

def preprocess_data(df):
    logging.info("Memulai proses data preprocessing...")
    
    # Hapus missing values & duplicate
    df = df.dropna().drop_duplicates()

    if 'Response' in df.columns:
        y = df['Response']
        X = df.drop(columns=['Response'])

        # 1. ENCODING: Ubah data teks/kategori menjadi angka (One-Hot Encoding)
        cat_cols = X.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            # Pastikan hasil True/False dari pandas versi baru berubah jadi 1.0 / 0.0
            X = X.astype(float) 
            logging.info(f"Kolom teks berhasil diubah menjadi angka: {list(cat_cols)}")

        # 2. SCALING: Standarisasi nilai angka
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        # Gabungkan kembali
        df_clean = X.copy()
        df_clean['Response'] = y.values
    else:
        df_clean = df.copy()

    return df_clean

if __name__ == "__main__":
    # Path dibaca dari posisi terminal root (msml-dicoding-submission)
    INPUT_PATH = 'Eksperimen_SML_valenisaa/dataset_raw.csv'
    OUTPUT_PATH = 'Workflow-CI/MLProject/dataset_preprocessing.csv'
    
    try:
        raw_data = load_data(INPUT_PATH)
        clean_data = preprocess_data(raw_data)
        
        # Simpan file bersih
        clean_data.to_csv(OUTPUT_PATH, index=False)
        logging.info(f"Data bersih yang siap masuk model berhasil disimpan ke {OUTPUT_PATH}")
    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")