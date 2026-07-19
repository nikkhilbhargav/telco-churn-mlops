# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements first
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 5. Copy the complete project
COPY . .

# Python settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]