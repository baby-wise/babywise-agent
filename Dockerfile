FROM python:3.11-slim AS base

WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar solo requirements primero para aprovechar cache
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la app
COPY . .

# Comando por defecto
CMD ["python", "agent.py", "start"]
