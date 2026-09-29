FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DERIVBOT_HOST=0.0.0.0

WORKDIR /srv/derivbot
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY research ./research
COPY scripts ./scripts
COPY config.example.yaml ./config.example.yaml

# Non-root user; state lives in mounted volumes.
RUN useradd --create-home --uid 10001 derivbot \
    && mkdir -p /srv/derivbot/data /srv/derivbot/logs /srv/derivbot/model \
    && chown -R derivbot:derivbot /srv/derivbot
USER derivbot

# Inside the container the app listens on 0.0.0.0:8000; docker-compose publishes it on 127.0.0.1 only.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

CMD ["python", "-m", "app.main"]
