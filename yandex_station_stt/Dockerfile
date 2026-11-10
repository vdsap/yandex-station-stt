FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (glibc prebuilt wheels available for 3.12)
RUN pip install --upgrade pip && \
    pip install \
    ai-edge-litert>=2.0.0 \
    numpy>=1.24.0 \
    wyoming>=1.5.0

# Copy application files
COPY yandex_stt /app/yandex_stt
COPY models /app/models
COPY run.sh /app/run.sh

RUN chmod +x /app/run.sh

EXPOSE 10300

ENTRYPOINT ["/app/run.sh"]
