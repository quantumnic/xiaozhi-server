FROM python:3.12-slim

# Install opus library for audio codec
RUN apt-get update && \
    apt-get install -y --no-install-recommends libopus0 libopus-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY . .

EXPOSE 8765

CMD ["python", "server.py"]
