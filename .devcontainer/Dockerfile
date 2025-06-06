FROM python:3.12.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y git wget \
 && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt