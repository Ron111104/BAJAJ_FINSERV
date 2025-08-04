# Use a lightweight base image
FROM python:3.11-slim

# Set environment variable to prevent .pyc files and buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (required for PyMuPDF and FAISS)
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Expose port for the web service
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
