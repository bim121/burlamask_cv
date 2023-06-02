FROM python:3.11

WORKDIR /app

# Установка системных зависимостей для dlib
RUN apt-get update && \
    apt-get install -y build-essential cmake && \
    apt-get install -y libopenblas-dev liblapack-dev && \
    apt-get install -y libx11-dev libgtk-3-dev

# Установка зависимостей Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование остальных файлов проекта
COPY . .

# Установка dlib в Windows
RUN pip install --no-cache-dir dlib==19.24.1 --install-option="--yes --compile --no-binary :all:"

CMD ["python", "burlamask.py"]
