# Gunakan image Python ringan
FROM python:3.14-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh file proyek ke dalam container
COPY . .

# Expose port 5001 untuk API model
EXPOSE 5001

# Command untuk menyalakan MLflow Model Serving
CMD ["mlflow", "models", "serve", "-m", "models/mlflow_model", "-h", "0.0.0.0", "-p", "5001", "--env-manager=local"]