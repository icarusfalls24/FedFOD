FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set noninteractive environment variable
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV, CLIP, and general build requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports: FastAPI (8000), Flower gRPC Coordinator (8082)
EXPOSE 8000
EXPOSE 8082

# Default command starts the API bridge server
CMD ["python", "api_server.py"]
