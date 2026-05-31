

import pandas as pd
import os
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def load_data(file_path):
    logging.info(f"Memuat data dari {file_path}")

    # Membaca dataset
    df = pd.read_csv(file_path, sep='\t')

    logging.info(f"Dataset berhasil dimuat dengan shape: {df.shape}")

    return df


def preprocess_data(df):

    logging.info("Memulai proses preprocessing data...")



    preprocessing_df = df.copy()


    if 'ID' in preprocessing_df.columns:
        preprocessing_df.drop(columns=['ID'], inplace=True)
        logging.info("Kolom ID berhasil dihapus.")


    current_year = 2025

    if 'Year_Birth' in preprocessing_df.columns:
        preprocessing_df['Age'] = current_year - preprocessing_df['Year_Birth']
        logging.info("Feature Age berhasil dibuat.")

    spending_columns = [
        'MntWines',
        'MntFruits',
        'MntMeatProducts',
        'MntFishProducts',
        'MntSweetProducts',
        'MntGoldProds'
    ]

    existing_spending_cols = [
        col for col in spending_columns
        if col in preprocessing_df.columns
    ]

    if existing_spending_cols:
        preprocessing_df['TotalSpending'] = preprocessing_df[
            existing_spending_cols
        ].sum(axis=1)

        logging.info("Feature TotalSpending berhasil dibuat.")



    numeric_cols = preprocessing_df.select_dtypes(
        include=['int64', 'float64']
    ).columns

    for col in numeric_cols:
        preprocessing_df[col].fillna(
            preprocessing_df[col].median(),
            inplace=True
        )

    logging.info("Missing value numerik berhasil ditangani.")

    categorical_cols = preprocessing_df.select_dtypes(
        include=['object']
    ).columns

    for col in categorical_cols:
        preprocessing_df[col].fillna(
            preprocessing_df[col].mode()[0],
            inplace=True
        )

    logging.info("Missing value kategorikal berhasil ditangani.")



    before_duplicate = preprocessing_df.shape[0]

    preprocessing_df.drop_duplicates(inplace=True)

    after_duplicate = preprocessing_df.shape[0]

    logging.info(
        f"Duplikat berhasil dihapus: {before_duplicate - after_duplicate}"
    )


    label_encoder = LabelEncoder()

    for col in categorical_cols:
        preprocessing_df[col] = label_encoder.fit_transform(
            preprocessing_df[col]
        )

    logging.info("Encoding data kategorikal berhasil dilakukan.")


    X = preprocessing_df.drop('Response', axis=1)
    y = preprocessing_df['Response']

    logging.info("Pemisahan feature dan target berhasil.")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    logging.info("Feature scaling berhasil dilakukan.")



    processed_df = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    processed_df['Response'] = y.values

    logging.info("Dataset preprocessing berhasil dibuat.")

    return processed_df



def save_data(df, output_path):

    # Membuat folder jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Simpan dataset
    df.to_csv(output_path, index=False)

    logging.info(
        f"Dataset preprocessing berhasil disimpan ke {output_path}"
    )



if __name__ == "__main__":

    # Path dataset input
    INPUT_PATH = os.path.join(
        os.path.dirname(__file__),
        '../data/raw/dataset_raw.csv'
    )

    # Path dataset output
    OUTPUT_PATH = os.path.join(
        os.path.dirname(__file__),
        '../data/processed/dataset_clean.csv'
    )

    try:

        # Load data
        raw_data = load_data(INPUT_PATH)

        # Preprocessing
        clean_data = preprocess_data(raw_data)

        # Save data
        save_data(clean_data, OUTPUT_PATH)

        logging.info(
            "Yeay! Pipeline preprocessing berhasil dijalankan."
        )

    except Exception as e:

        logging.error(f"Terjadi kesalahan: {e}")

