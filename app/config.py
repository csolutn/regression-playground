import os
from datetime import timedelta


class Config:
    APP_NAME = 'Machine Learning Playground'   # shown in every language
    TEACHER_NAME = os.environ.get('TEACHER_NAME', '정솔')   # login page: whom to ask about a lost password
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///playground.db')   # file in instance/
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_HTTPONLY = True
    # behind a proxy that ends HTTPS (cloudflared in compose.yaml): trust its X-Forwarded-* and send secure cookies
    TRUST_PROXY = os.environ.get('TRUST_PROXY') == '1'
    SESSION_COOKIE_SECURE = TRUST_PROXY
    PERMANENT_SESSION_LIFETIME = timedelta(hours=12)
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024
    # worker processes that train at once (one CPU core each); other runs wait in line
    MAX_CONCURRENT_TRAININGS = int(os.environ.get('MAX_CONCURRENT_TRAININGS', min(6, os.cpu_count() or 2)))
    QUEUE_TIMEOUT_S = float(os.environ.get('QUEUE_TIMEOUT_S', 90))              # then "server is busy"
    TRAINING_TIME_LIMIT_S = float(os.environ.get('TRAINING_TIME_LIMIT_S', 30))  # then stopped, saved as 'time limit'
    BABEL_DEFAULT_LOCALE = 'ko'
    BABEL_TRANSLATION_DIRECTORIES = 'translations'
