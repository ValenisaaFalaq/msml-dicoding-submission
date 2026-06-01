# msml-dicoding-submission
# Customer Personality Analysis - End-to-End MLOps Pipeline

Proyek ini merupakan submission untuk kelas "Membangun Sistem Machine Learning" dari Dicoding. Repositori ini mendemonstrasikan alur kerja MLOps (Machine Learning Operations) skala industri, mencakup otomatisasi pra-pemrosesan data, pelacakan eksperimen berbasis cloud, hingga deployment dan monitoring menggunakan container.

## Arsitektur Sistem & Teknologi

* **Bahasa Pemrograman:** Python 3.10
* **Machine Learning:** Scikit-Learn (RandomForestClassifier)
* **Hyperparameter Tuning:** GridSearchCV
* **Experiment Tracking:** MLflow & DagsHub (Manual Logging)
* **Containerization & Serving:** Docker & Docker Compose
* **System Monitoring:** Prometheus & Grafana

## Alur Eksperimen (Pipeline)

1.  **Automated Preprocessing:** Pembersihan data mentah, penanganan *missing values*, dan transformasi data dilakukan secara otomatis melalui *script* `src/automate_preprocessing.py` yang dieksekusi melalui GitHub Actions.
2.  **Modelling & Tuning:** Pencarian parameter model terbaik dicatat secara ketat menggunakan metode *manual logging* pada MLflow, mencakup pencatatan parameter, metrik, dan artefak visual (Confusion Matrix & Classification Report).
3.  **Model Serving:** Model klasifikasi terbaik diekspor ke dalam format MLflow Model dan disajikan sebagai REST API yang berjalan secara independen di dalam Docker Container.
4.  **Real-time Monitoring:** Status kesehatan lingkungan sistem (System Environment Health) dipantau terus-menerus oleh Prometheus dan divisualisasikan melalui *dashboard* interaktif Grafana.

## Tautan Eksperimen DagsHub

🌐 **MLflow Tracking Dashboard:** [https://dagshub.com/ValenisaaFalaq/msml-dicoding-submission.mlflow]

## Struktur Repositori

```text
msml-dicoding-submission/
├── .github/workflows/         # Konfigurasi GitHub Actions CI/CD
├── assets/                    # Penyimpanan tangkapan layar bukti monitoring
├── data/                      # Direktori dataset mentah dan bersih
├── models/mlflow_model/       # Artefak model berformat MLflow
├── monitoring/                # Konfigurasi Prometheus
├── src/
│   ├── artifacts/             # Laporan teks dan gambar evaluasi model
│   ├── automate_preprocessing.py
│   ├── modelling.py
│   └── modelling_tuning.py
├── MLproject                  # Konfigurasi MLflow run environment
├── conda.yaml                 # Spesifikasi environment dependency
├── docker-compose.yml         # Orkestrasi layanan Model, Prometheus, dan Grafana
├── Dockerfile                 # Resep pembuatan container Model Serving
└── requirements.txt           # Daftar pustaka Python
