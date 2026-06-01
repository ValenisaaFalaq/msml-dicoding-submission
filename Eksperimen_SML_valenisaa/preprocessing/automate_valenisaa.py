import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    logging.info(f"Memuat data dari {file_path}")
    return pd.read_csv(file_path, sep='\t') # Gunakan sep='\t' jika file aslinya terpisahkan oleh tab, hapus parameter ini jika pakai koma.

def preprocess_data(df):
    logging.info("Memulai proses data preprocessing...")
    
    # 1. Hapus nilai kosong (Missing Values)
    df = df.dropna()
    logging.info("Missing values berhasil dihapus.")
    
    # 2. Hapus data duplikat
    df = df.drop_duplicates()
    logging.info("Data duplikat berhasil dihapus.")

    # 3. Pisahkan Fitur dan Target (Response)
    if 'Response' in df.columns:
        y = df['Response']
        X = df.drop(columns=['Response'])
        
        # 4. Ambil hanya kolom angka (numerik) untuk di-scaling agar tidak error saat baca teks
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # 5. Lakukan Scaling
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        
        # 6. Gabungkan kembali data yang sudah bersih
        df_clean = X.copy()
        df_clean['Response'] = y.values
    else:
        df_clean = df.copy()
        logging.warning("Kolom 'Response' tidak ditemukan, melewati tahap pemisahan target.")

    logging.info("Data preprocessing selesai.")
    return df_clean

def save_data(df, output_path):
    df.to_csv(output_path, index=False)
    logging.info(f"Data bersih berhasil disimpan ke {output_path}")

if __name__ == "__main__":
    # Mengatur rute file otomatis berdasarkan posisi folder tempat file ini berada
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Dataset mentah ada di luar folder preprocessing
    INPUT_PATH = os.path.join(BASE_DIR, '../dataset_raw.csv')
    
    # Dataset bersih akan disimpan di dalam folder preprocessing
    OUTPUT_PATH = os.path.join(BASE_DIR, 'dataset_preprocessing.csv')
    
    # Eksekusi Alur
    try:
        # Note: Jika dataset_raw.csv kamu dipisahkan oleh koma biasa, 
        # kamu bisa menghapus parameter sep='\t' pada fungsi load_data di atas.
        raw_data = load_data(INPUT_PATH)
        clean_data = preprocess_data(raw_data)
        save_data(clean_data, OUTPUT_PATH)
        logging.info("Yeay! Pipeline preprocessing otomatis berhasil dijalankan.")
    except Exception as e:
        logging.error(f"Terjadi kesalahan: {e}")