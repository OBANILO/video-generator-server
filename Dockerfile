FROM python:3.11-slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg fonts-liberation fonts-dejavu fonts-freefont-ttf && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --timeout 600"]
