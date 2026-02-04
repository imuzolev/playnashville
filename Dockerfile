FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
RUN mkdir -p /app/results && useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]
