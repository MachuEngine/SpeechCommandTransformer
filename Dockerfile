FROM python:3.9-slim

WORKDIR /app

# System deps for torchaudio sox backend
RUN apt-get update \
    && apt-get install -y --no-install-recommends sox libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TORCHAUDIO_BACKEND=sox_io

COPY requirements.txt /app/requirements.txt
COPY requirements-server.txt /app/requirements-server.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements-server.txt

# Copy the rest of the project (excluded by .dockerignore)
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

