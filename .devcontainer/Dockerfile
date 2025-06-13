FROM python:3.12.11-slim
#TODO: Configurer Git pour éviter de ressaisir les identifiants dans le Dev Container et en mode container du docker-compose
WORKDIR /workspace

RUN apt-get update && apt-get install -y git wget \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --timeout 600 tensorflow==2.19.0

COPY requirements.txt .
# Augmentez le timeout à 600 secondes (10 minutes) pour fix bug "pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out."
RUN pip install --no-cache-dir --timeout 600 -r requirements.txt