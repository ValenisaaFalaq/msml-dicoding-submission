import requests

# URL endpoint model serving MLflow (sesuaikan dengan port-mu)
url = "http://127.0.0.1:5001/ping"

try:
    response = requests.get(url)
    if response.status_code == 200:
        print("✅ Model Serving Aktif! Status Code:", response.status_code)
    else:
        print("❌ Gagal menjangkau model. Status Code:", response.status_code)
except Exception as e:
    print("❌ Terjadi error:", str(e))