# Use Python 3.10 slim image
FROM python:3.10-slim

# Install needed system libs for TensorFlow & image processing
RUN apt-get update && apt-get install -y \
     libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
     && apt-get clean

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Cloud Run expects PORT env
ENV PORT=8080

# Run FastAPI on Uvicorn
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "8080"]
