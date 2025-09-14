FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

# Ставим torch заранее с правильным wheel
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Ставим остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
