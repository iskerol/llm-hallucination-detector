FROM python:3.10-slim

# Enforce non-interactive execution constraints
ENV DEBIAN_FRONTEND=noninteractive

# Update system topologies enabling basic FAISS C++ mapping dependencies bounds
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement boundaries optimizing Docker layers caching loops
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Transport core structures arrays
COPY . .

# Initialize application space topologies locally natively
RUN pip install -e .

EXPOSE 8000
EXPOSE 7860

# Background execution multiplex shell block arrays
RUN echo '#!/bin/bash\nuvicorn api.main:app --host 0.0.0.0 --port 8000 & \npython ui/app.py \nwait -n' > /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
