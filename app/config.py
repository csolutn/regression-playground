import os
from datetime import timedelta


class Config:
    APP_NAME = 'ŷ Playground'   # shown in every language
    # the credits under the playground (each one left out when empty): '© <CREDIT_NAME>' linking to CREDIT_URL,
    # a GitHub icon, an e-mail icon
    CREDIT_NAME = os.environ.get('CREDIT_NAME', '')
    CREDIT_URL = os.environ.get('CREDIT_URL', '')
    GITHUB_URL = os.environ.get('GITHUB_URL', '')
    CONTACT_EMAIL = os.environ.get('CONTACT_EMAIL', '')
    # the first teacher account, made at start-up while there is no teacher yet (later: flask create-teacher)
    ADMIN_LOGIN_ID = os.environ.get('ADMIN_LOGIN_ID', '')
    ADMIN_NAME = os.environ.get('ADMIN_NAME', '')
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', '')
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
