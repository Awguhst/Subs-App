# Bulgarian Subtitle API: FastAPI + FFmpeg. Works on Railway as-is (listens on $PORT).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/app backend/app
COPY frontend/index.html frontend/index.html
COPY --chmod=0755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint
# Tolerate a Windows checkout that converted the script to CRLF line endings.
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint

WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", \"8000\")}/health', timeout=4)"
ENTRYPOINT ["docker-entrypoint"]
CMD ["python", "-m", "app"]
