FROM python:3.11

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential cmake && \
    apt-get install -y libopenblas-dev liblapack-dev && \
    apt-get install -y libx11-dev libgtk-3-dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir dlib==19.24.1 --install-option="--yes --compile --no-binary :all:"

CMD ["python", "burlamask.py"]
