# Production image: gunicorn (one process, many threads) + training worker processes.
# Build and run with compose.yaml; see "Deploy" in README.md.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONUNBUFFERED=1 PATH=/opt/venv/bin:$PATH HOME=/tmp

WORKDIR /srv
COPY pyproject.toml uv.lock ./
RUN UV_NO_CACHE=1 uv sync --locked --no-dev --no-install-project      # no download cache left in the image
COPY app ./app

# runs as a plain user; only instance/ (the SQLite database, a volume) is writable
RUN useradd --uid 10001 --no-create-home app && mkdir instance && chown app instance
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/login', timeout=4)"
CMD ["gunicorn", "--workers", "1", "--threads", "40", "--bind", "0.0.0.0:8000", \
     "--timeout", "120", "--graceful-timeout", "40", "--access-logfile", "-", "app:create_app()"]
