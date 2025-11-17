# Use a stable slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better build caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project (app.py, models/, etc.)
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
